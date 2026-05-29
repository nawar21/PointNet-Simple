"""
PointNet, from scratch.

This file implements the architecture from
    Qi et al., "PointNet: Deep Learning on Point Sets for 3D Classification
    and Segmentation" (CVPR 2017).

The whole paper rests on ONE idea (Eq. 1):

        f({x_1, ..., x_n})  ~=  gamma( MAX_i { h(x_i) } )

  - h  is a shared MLP applied INDEPENDENTLY to every point (implemented as
       1x1 convolutions, so the same weights touch each point),
  - MAX is a symmetric function (max-pool over points) -> this is what makes
       the network invariant to the ordering of the input points,
  - gamma is a small MLP that turns the pooled global feature into scores.

The optional T-Nets predict little affine matrices that re-align the input
(and, later, the features) into a canonical pose before processing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def shared_mlp(channels):
    """A stack of shared per-point layers: Conv1d(1x1) + BatchNorm + ReLU.

    A 1x1 conv over the point dimension is exactly 'apply the same MLP to
    every point independently' -- the weight sharing the paper relies on.
    """
    layers = []
    for c_in, c_out in zip(channels[:-1], channels[1:]):
        layers += [nn.Conv1d(c_in, c_out, 1), nn.BatchNorm1d(c_out), nn.ReLU()]
    return nn.Sequential(*layers)


class TNet(nn.Module):
    """Mini-PointNet that regresses a (k x k) alignment matrix (Fig. 2)."""

    def __init__(self, k):
        super().__init__()
        self.k = k
        self.feat = shared_mlp([k, 64, 128, 1024])
        self.fc = nn.Sequential(
            nn.Linear(1024, 512), nn.BatchNorm1d(512), nn.ReLU(),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(),
            nn.Linear(256, k * k),
        )

    def forward(self, x):                 # x: (B, k, N)
        B = x.size(0)
        h = self.feat(x)                  # (B, 1024, N)
        h = torch.max(h, dim=2)[0]        # symmetric pool -> (B, 1024)
        m = self.fc(h)                    # (B, k*k)
        # start from identity so an untrained T-Net does nothing harmful
        eye = torch.eye(self.k, device=x.device).flatten()
        m = m + eye
        return m.view(B, self.k, self.k)


class PointNetEncoder(nn.Module):
    """Maps a point cloud -> a single global feature vector.

    Returns the global feature, the feature-alignment matrix (for the
    regularization loss), and the argmax indices from the max-pool -- those
    indices identify the *critical points* (Theorem 2 in the paper).
    """

    def __init__(self, global_dim=1024, use_tnet=True):
        super().__init__()
        self.use_tnet = use_tnet
        self.input_tnet = TNet(3) if use_tnet else None
        self.mlp1 = shared_mlp([3, 64, 64])
        self.feat_tnet = TNet(64) if use_tnet else None
        self.mlp2 = shared_mlp([64, 128, global_dim])

    def forward(self, x):                 # x: (B, N, 3)
        x = x.transpose(1, 2)             # (B, 3, N) for Conv1d
        if self.use_tnet:
            t_in = self.input_tnet(x)                      # (B, 3, 3)
            x = torch.bmm(t_in.transpose(1, 2), x)         # align input points

        x = self.mlp1(x)                                   # (B, 64, N)

        feat_mat = None
        if self.use_tnet:
            feat_mat = self.feat_tnet(x)                   # (B, 64, 64)
            x = torch.bmm(feat_mat.transpose(1, 2), x)     # align features

        x = self.mlp2(x)                                   # (B, global_dim, N)
        global_feat, critical_idx = torch.max(x, dim=2)    # (B, D), (B, D)
        return global_feat, feat_mat, critical_idx


class PointNetClassifier(nn.Module):
    """Full classification network (Fig. 2, top branch)."""

    def __init__(self, num_classes, global_dim=1024, use_tnet=True):
        super().__init__()
        self.encoder = PointNetEncoder(global_dim, use_tnet)
        self.head = nn.Sequential(
            nn.Linear(global_dim, 512), nn.BatchNorm1d(512), nn.ReLU(),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        global_feat, feat_mat, critical_idx = self.encoder(x)
        logits = self.head(global_feat)
        return logits, feat_mat, critical_idx


def feature_transform_regularizer(feat_mat):
    """L_reg = || I - A A^T ||_F^2   (Eq. 2): keep the 64x64 matrix orthogonal."""
    if feat_mat is None:
        return torch.tensor(0.0)
    B, k, _ = feat_mat.size()
    eye = torch.eye(k, device=feat_mat.device).unsqueeze(0)
    prod = torch.bmm(feat_mat, feat_mat.transpose(1, 2))
    return torch.mean(torch.norm(eye - prod, dim=(1, 2)) ** 2)
