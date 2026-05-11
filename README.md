# Trustworthy AI Assignment #3 — Marabou Neural Network Verification

This project demonstrates the use of [Marabou](https://github.com/NeuralNetworkVerification/Marabou)
to verify local robustness properties of a small MNIST classifier.

## Setup

Requires **Python 3.11** (maraboupy only supports 3.8–3.11).

```bash
conda create -n marabou python=3.11 -y
conda activate marabou
pip install -r requirements.txt
```

## Usage

```bash
# 1. Train the model and export to ONNX
python scripts/train_mnist.py

# 2. Run the verification experiments
python test.py
```

## Project structure
.
├── scripts/
│   ├── train_mnist.py     # Train FC network and export to ONNX
│   └── verify.py          # Run Marabou verification queries
├── models/                # Trained ONNX model
├── results/               # Verification logs and counterexample images
├── test.py                # End-to-end verification demo
├── report.pdf             # Analysis report (1–2 pages)
└── requirements.txt
