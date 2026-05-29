"""
Critical points -- visualizing the paper's most striking result.

Theorem 2 says the global feature f(S) is determined by a small subset of the
input: the *critical point set*. These are exactly the points that "win" the
max-pool for at least one feature dimension. The paper shows (Fig. 7) that this
sparse subset forms the SKELETON of the object.

Our encoder already returns the argmax indices from the max-pool, so finding the
critical set is a one-liner. We render, for each shape, the full cloud next to
its critical points -- and report how few points actually drive the prediction.

    python -m demos.critical_points

Saves: assets/critical_points.png
"""
import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pointnet import CLASSES, make_shape
from pointnet.model import PointNetClassifier


def critical_indices(model, pts):
    """Return the unique input indices that drive the max-pooled global feature."""
    with torch.no_grad():
        x = torch.from_numpy(pts).unsqueeze(0)
        _, _, critical_idx = model.encoder(x)     # (1, global_dim)
    return np.unique(critical_idx.numpy().ravel())


def main(checkpoint="pointnet.pt", n_points=1024, seed=7,
         out="assets/critical_points.png"):
    model = PointNetClassifier(len(CLASSES))
    try:
        model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    except FileNotFoundError:
        print("warning: no checkpoint -- critical sets are most meaningful "
              "after training. Run demos.train_classifier first.")
    model.eval()

    rng = np.random.default_rng(seed)
    fig = plt.figure(figsize=(3 * len(CLASSES), 6))
    for col, label in enumerate(range(len(CLASSES))):
        pts = make_shape(label, n_points, rng)
        cidx = critical_indices(model, pts)
        frac = len(cidx) / n_points

        for row, (subset, title, color) in enumerate([
            (pts, f"{CLASSES[label]}\nall {n_points} pts", "#9aa7b8"),
            (pts[cidx], f"critical: {len(cidx)} pts\n({frac:.0%})", "#e8590c"),
        ]):
            ax = fig.add_subplot(2, len(CLASSES), row * len(CLASSES) + col + 1,
                                 projection="3d")
            ax.scatter(subset[:, 0], subset[:, 1], subset[:, 2],
                       s=6 if row == 0 else 14, c=color, depthshade=True)
            ax.set_title(title, fontsize=9)
            ax.set_axis_off()
            ax.set_box_aspect((1, 1, 1))

    fig.suptitle("Critical point sets — the sparse 'skeleton' that determines "
                 "the global feature (PointNet, Thm. 2)", fontsize=12)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out, dpi=130)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
