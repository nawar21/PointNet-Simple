"""
Synthetic 3D point-cloud dataset.

The PointNet paper trains on ModelNet40 / ShapeNet, which are large downloads.
To keep this repo *runnable in seconds with no downloads*, we procedurally
generate point clouds for six simple geometric primitives. The shapes are
distinct enough to train fast, yet share the exact properties that make point
clouds hard (they are unordered, vary in point count, and live in continuous
3D space) -- so every idea in the paper still applies.

Each sample is:
  - a set of N points sampled on a shape's surface,
  - mean-centered and scaled into the unit sphere (as the paper does),
  - lightly jittered with Gaussian noise (the paper's training augmentation).
"""

import numpy as np
import torch
from torch.utils.data import Dataset

CLASSES = ["sphere", "cube", "cylinder", "cone", "torus", "plane"]


def _normalize(pts: np.ndarray) -> np.ndarray:
    """Center at origin and scale so the farthest point sits on the unit sphere."""
    pts = pts - pts.mean(axis=0, keepdims=True)
    scale = np.max(np.linalg.norm(pts, axis=1))
    return pts / (scale + 1e-8)


def _sphere(n, rng):
    v = rng.normal(size=(n, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True) + 1e-8
    return v


def _cube(n, rng):
    # pick a random face for each point, then a random position on that face
    pts = rng.uniform(-1, 1, size=(n, 3))
    axis = rng.integers(0, 3, size=n)
    sign = rng.choice([-1.0, 1.0], size=n)
    pts[np.arange(n), axis] = sign
    return pts


def _cylinder(n, rng):
    n_side = int(n * 0.7)
    n_cap = n - n_side
    theta = rng.uniform(0, 2 * np.pi, n_side)
    z = rng.uniform(-1, 1, n_side)
    side = np.stack([np.cos(theta), np.sin(theta), z], axis=1)
    r = np.sqrt(rng.uniform(0, 1, n_cap))
    phi = rng.uniform(0, 2 * np.pi, n_cap)
    cz = rng.choice([-1.0, 1.0], n_cap)
    cap = np.stack([r * np.cos(phi), r * np.sin(phi), cz], axis=1)
    return np.concatenate([side, cap], axis=0)


def _cone(n, rng):
    # lateral surface: radius shrinks linearly from base (z=-1) to tip (z=1)
    z = rng.uniform(-1, 1, n)
    radius = (1 - z) / 2.0
    theta = rng.uniform(0, 2 * np.pi, n)
    return np.stack([radius * np.cos(theta), radius * np.sin(theta), z], axis=1)


def _torus(n, rng, R=0.7, r=0.3):
    u = rng.uniform(0, 2 * np.pi, n)
    v = rng.uniform(0, 2 * np.pi, n)
    x = (R + r * np.cos(v)) * np.cos(u)
    y = (R + r * np.cos(v)) * np.sin(u)
    z = r * np.sin(v)
    return np.stack([x, y, z], axis=1)


def _plane(n, rng):
    xy = rng.uniform(-1, 1, size=(n, 2))
    z = np.zeros((n, 1))
    return np.concatenate([xy, z], axis=1)


_GENERATORS = {
    "sphere": _sphere,
    "cube": _cube,
    "cylinder": _cylinder,
    "cone": _cone,
    "torus": _torus,
    "plane": _plane,
}


def make_shape(label: int, n_points: int, rng: np.random.Generator,
               jitter: float = 0.02) -> np.ndarray:
    """Generate one normalized, jittered point cloud for a class index."""
    pts = _GENERATORS[CLASSES[label]](n_points, rng)
    pts = _normalize(pts)
    pts = pts + rng.normal(0, jitter, size=pts.shape)  # paper uses std 0.02
    return pts.astype(np.float32)


class ShapeDataset(Dataset):
    """A reproducible synthetic point-cloud classification dataset."""

    def __init__(self, n_samples=1200, n_points=512, seed=0, jitter=0.02):
        self.n_points = n_points
        self.jitter = jitter
        rng = np.random.default_rng(seed)
        self.labels = rng.integers(0, len(CLASSES), size=n_samples)
        # one RNG seed per sample -> deterministic but unique clouds
        self.seeds = rng.integers(0, 2**31 - 1, size=n_samples)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        rng = np.random.default_rng(int(self.seeds[idx]))
        label = int(self.labels[idx])
        pts = make_shape(label, self.n_points, rng, self.jitter)
        return torch.from_numpy(pts), label
