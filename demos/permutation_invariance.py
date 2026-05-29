"""
Permutation invariance -- the central claim of the paper, tested empirically.

A point cloud is a *set*: shuffling the order of its points must not change the
prediction. We feed the network the SAME cloud under many random orderings and
check that the global feature and the class scores are identical (up to
floating-point noise). We also feed a SORTED ordering to hammer the point home.

    python -m demos.permutation_invariance
"""
import numpy as np
import torch

from pointnet import CLASSES, make_shape
from pointnet.model import PointNetClassifier


def main(checkpoint="pointnet.pt", n_points=512, n_perms=8, seed=1):
    model = PointNetClassifier(len(CLASSES))
    try:
        model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
        print(f"loaded {checkpoint}")
    except FileNotFoundError:
        print("no checkpoint found -- using a random (untrained) net; "
              "invariance is an architectural property and holds regardless.")
    model.eval()

    rng = np.random.default_rng(seed)
    label = rng.integers(0, len(CLASSES))
    pts = make_shape(label, n_points, rng)              # (N, 3)
    print(f"\ntrue shape: {CLASSES[label]}\n")

    feats, preds = [], []
    orderings = ["original", "sorted"] + [f"shuffle {i}" for i in range(n_perms)]
    for name in orderings:
        if name == "original":
            p = pts
        elif name == "sorted":
            order = np.lexsort((pts[:, 2], pts[:, 1], pts[:, 0]))
            p = pts[order]
        else:
            p = pts[rng.permutation(n_points)]
        with torch.no_grad():
            x = torch.from_numpy(p).unsqueeze(0)
            logits, _, _ = model.forward(x)
            gfeat, _, _ = model.encoder(x)
        feats.append(gfeat.numpy().ravel())
        preds.append(CLASSES[logits.argmax(1).item()])
        print(f"  {name:>10s}  ->  predicted {preds[-1]:9s}  "
              f"logit_max={logits.max().item():+.4f}")

    feats = np.stack(feats)
    max_dev = np.abs(feats - feats[0]).max()
    print(f"\nmax deviation of the 1024-dim global feature across "
          f"{len(orderings)} orderings: {max_dev:.2e}")
    print("all predictions identical:", len(set(preds)) == 1)
    print("\n=> The ordering of the input points does not matter. "
          "Max-pooling makes the network a true set function.")


if __name__ == "__main__":
    main()
