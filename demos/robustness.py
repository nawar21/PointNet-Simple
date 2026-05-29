"""
Robustness to missing data -- reproducing the idea behind Fig. 6 (left).

Because the prediction depends only on a sparse set of critical points, dropping
a large fraction of the input should barely move the accuracy. We progressively
delete points from the test set and plot accuracy vs. the missing-data ratio.

    python -m demos.robustness

Saves: assets/robustness.png
"""
import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pointnet import CLASSES, ShapeDataset
from pointnet.model import PointNetClassifier


@torch.no_grad()
def accuracy_with_dropout(model, dataset, keep_ratio, seed=0, batch_size=64):
    """For a fixed keep_ratio every cloud keeps the same number of points,
    so we can drop points per-cloud and still evaluate in batches."""
    rng = np.random.default_rng(seed)
    clouds, labels = [], []
    for pts, label in dataset:
        n = pts.shape[0]
        k = max(1, int(round(n * keep_ratio)))
        idx = rng.choice(n, size=k, replace=False)
        clouds.append(pts[idx])
        labels.append(label)

    correct = 0
    for i in range(0, len(clouds), batch_size):
        x = torch.stack(clouds[i:i + batch_size])
        y = torch.tensor(labels[i:i + batch_size])
        logits, _, _ = model(x)
        correct += (logits.argmax(1) == y).sum().item()
    return correct / len(clouds)


def main(checkpoint="pointnet.pt", out="assets/robustness.png"):
    model = PointNetClassifier(len(CLASSES))
    try:
        model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    except FileNotFoundError:
        print("error: train a model first (python -m demos.train_classifier)")
        return
    model.eval()

    test = ShapeDataset(n_samples=180, n_points=256, seed=12345)
    missing = [0.0, 0.25, 0.5, 0.7, 0.85, 0.9, 0.95]
    accs = [accuracy_with_dropout(model, test, 1 - m) for m in missing]

    for m, a in zip(missing, accs):
        print(f"  missing {m:>4.0%}  ->  accuracy {a:.3f}")

    os.makedirs(os.path.dirname(out), exist_ok=True)
    plt.figure(figsize=(6, 4))
    plt.plot([m * 100 for m in missing], [a * 100 for a in accs],
             "o-", color="#1c7ed6")
    plt.xlabel("Missing data ratio (%)")
    plt.ylabel("Accuracy (%)")
    plt.title("PointNet stays accurate as points vanish")
    plt.grid(alpha=0.3)
    plt.ylim(0, 105)
    plt.tight_layout()
    plt.savefig(out, dpi=130)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
