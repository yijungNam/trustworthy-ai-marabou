"""
End-to-end Marabou verification demo for the MNIST FC classifier.

For each sample image we sweep over a range of perturbation radii (epsilons)
and ask Marabou: "Is the prediction provably unchanged under any L_inf
perturbation of size <= epsilon?"

We expect:
  - small epsilon  -> UNSAT  (robust, no counterexample exists)
  - large epsilon  -> SAT    (a counterexample is found)
  - somewhere in between is the empirical robustness boundary.

Outputs:
  results/verification_log.txt   - full per-query log
  results/summary.csv            - machine-readable summary
  results/counterexample_*.png   - visualizations of SAT inputs
"""

import csv
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")  # no GUI needed
import matplotlib.pyplot as plt

# Make the scripts/ directory importable
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from scripts.verify import verify_robustness  # noqa: E402


# --------------------------- Configuration ---------------------------
EPSILONS = [0.001, 0.005, 0.01, 0.02, 0.08, 0.1]
TIMEOUT_SEC = 100
N_IMAGES = 3  # verify the first N sample images

RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)


# --------------------------- Helpers ---------------------------
def save_counterexample(original, adversarial, true_label, adv_label,
                        epsilon, idx, out_path):
    """Save a 3-panel figure: original, adversarial, and pixel-wise diff."""
    fig, axes = plt.subplots(1, 3, figsize=(9, 3))

    axes[0].imshow(original.squeeze(), cmap="gray", vmin=0, vmax=1)
    axes[0].set_title(f"Original (label={true_label})")
    axes[0].axis("off")

    axes[1].imshow(adversarial.squeeze(), cmap="gray", vmin=0, vmax=1)
    axes[1].set_title(f"Adversarial (pred={adv_label})")
    axes[1].axis("off")

    diff = adversarial.squeeze() - original.squeeze()
    # symmetric color map centered at 0 so we see + and - perturbations
    vmax = max(abs(diff.min()), abs(diff.max()), 1e-6)
    im = axes[2].imshow(diff, cmap="seismic", vmin=-vmax, vmax=vmax)
    axes[2].set_title(f"Diff (epsilon={epsilon})")
    axes[2].axis("off")
    fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

    fig.suptitle(f"Counterexample for image #{idx}", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def log_and_print(line, fh):
    print(line)
    fh.write(line + "\n")
    fh.flush()


# --------------------------- Main ---------------------------
def main():
    images = np.load(ROOT / "models" / "sample_inputs.npy")
    labels = np.load(ROOT / "models" / "sample_labels.npy")

    summary_rows = []
    log_path = RESULTS_DIR / "verification_log.txt"
    overall_start = time.time()

    with open(log_path, "w") as fh:
        log_and_print(f"=== Marabou MNIST FC robustness sweep ===", fh)
        log_and_print(f"Images: {N_IMAGES} | Epsilons: {EPSILONS}", fh)
        log_and_print(f"Timeout per query: {TIMEOUT_SEC}s\n", fh)

        for img_idx in range(min(N_IMAGES, len(images))):
            image = images[img_idx]
            true_label = int(labels[img_idx])
            log_and_print(f"--- Image #{img_idx} (true label = {true_label}) ---", fh)

            for eps in EPSILONS:
                out = verify_robustness(image, true_label, eps, timeout=TIMEOUT_SEC)
                line = (f"  eps={eps:<6}  result={out['result']:<7} "
                        f"runtime={out['runtime']:7.2f}s")
                if out["adv_label"] is not None:
                    line += f"  adv_label={out['adv_label']}"
                log_and_print(line, fh)

                summary_rows.append({
                    "image_idx": img_idx,
                    "true_label": true_label,
                    "epsilon": eps,
                    "result": out["result"],
                    "runtime_sec": round(out["runtime"], 3),
                    "adv_label": out["adv_label"] if out["adv_label"] is not None else "",
                })

                # Save counterexample image whenever Marabou finds one
                if out["result"] == "SAT" and out["counterexample"] is not None:
                    fname = (f"counterexample_img{img_idx}_eps{eps}"
                             f"_true{true_label}_adv{out['adv_label']}.png")
                    save_counterexample(
                        image, out["counterexample"],
                        true_label, out["adv_label"], eps, img_idx,
                        RESULTS_DIR / fname,
                    )
                    log_and_print(f"    -> saved {fname}", fh)

            log_and_print("", fh)

        total = time.time() - overall_start
        log_and_print(f"Done in {total:.1f}s total.", fh)

    # CSV summary for the report
    csv_path = RESULTS_DIR / "summary.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"\nSummary written to {csv_path}")
    print(f"Full log written to {log_path}")


if __name__ == "__main__":
    main()
