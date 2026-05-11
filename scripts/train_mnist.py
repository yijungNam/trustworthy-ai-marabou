"""
Train a small fully-connected MNIST classifier and export it to ONNX.

Architecture:
    Flatten(28*28) -> Linear(128) -> ReLU -> Linear(64) -> ReLU -> Linear(10)

The network is intentionally kept small so Marabou can verify robustness
properties in a reasonable amount of time. Marabou natively supports
piecewise-linear activations (ReLU), so we avoid sigmoid/tanh.

Outputs:
    - models/mnist_fc.onnx : the trained model in ONNX format
    - models/sample_inputs.npy : a few test images used later for verification
    - models/sample_labels.npy : their ground-truth labels
"""

import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# ----------------------------- Model -----------------------------
class MnistFC(nn.Module):
    """Small fully-connected MNIST classifier (verifier-friendly)."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        # x: (N, 1, 28, 28) -> (N, 784)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # No softmax: Marabou verifies on the pre-softmax logits.
        return self.fc3(x)


# ----------------------------- Training utilities -----------------------------
def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        preds = model(images).argmax(dim=1)
        correct += (preds == labels).sum().item()
    return correct / len(loader.dataset)


# ----------------------------- Main -----------------------------
def main():
    here = Path(__file__).resolve().parent.parent  # project root
    models_dir = here / "models"
    models_dir.mkdir(exist_ok=True)
    data_dir = here / "data"

    # MNIST is normalized to [0, 1]. We keep the raw range so that an
    # epsilon perturbation in pixel space has an intuitive meaning.
    transform = transforms.Compose([transforms.ToTensor()])

    train_set = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    test_set = datasets.MNIST(data_dir, train=False, download=True, transform=transform)

    train_loader = DataLoader(train_set, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=512)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")

    model = MnistFC().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    EPOCHS = 3  # 3 epochs is enough to get ~97% accuracy on this small net
    for epoch in range(1, EPOCHS + 1):
        loss = train_one_epoch(model, train_loader, optimizer, device)
        acc = evaluate(model, test_loader, device)
        print(f"Epoch {epoch} | train loss = {loss:.4f} | test acc = {acc:.4f}")

    # ---------- Export to ONNX ----------
    # We use a single-image batch (N=1) so Marabou's ONNX parser sees a fixed
    # input shape. Marabou supports static shapes more reliably than dynamic.
    model.eval().cpu()
    dummy = torch.zeros(1, 1, 28, 28)
    onnx_path = models_dir / "mnist_fc.onnx"
    torch.onnx.export(
        model,
        dummy,
        onnx_path,
        input_names=["input"],
        output_names=["logits"],
        opset_version=12,  # opset 12 is well-supported by Marabou
        dynamo=False,        
    )
    print(f"Exported ONNX model to {onnx_path}")

    # ---------- Save sample test inputs for verification ----------
    # We will pick a handful of correctly-classified test images and reuse
    # them in test.py / verify.py.
    sample_images = []
    sample_labels = []
    with torch.no_grad():
        for images, labels in test_loader:
            preds = model(images).argmax(dim=1)
            for i in range(images.size(0)):
                if preds[i] == labels[i]:
                    sample_images.append(images[i].numpy())
                    sample_labels.append(int(labels[i].item()))
                if len(sample_images) >= 5:
                    break
            if len(sample_images) >= 5:
                break

    np.save(models_dir / "sample_inputs.npy", np.stack(sample_images))
    np.save(models_dir / "sample_labels.npy", np.array(sample_labels))
    print(f"Saved {len(sample_images)} sample inputs for verification")


if __name__ == "__main__":
    main()
