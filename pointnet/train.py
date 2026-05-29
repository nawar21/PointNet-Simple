"""Minimal training / evaluation loop for the classification PointNet."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .data import ShapeDataset, CLASSES
from .model import PointNetClassifier, feature_transform_regularizer


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    for pts, labels in loader:
        pts, labels = pts.to(device), labels.to(device)
        logits, _, _ = model(pts)
        correct += (logits.argmax(1) == labels).sum().item()
        total += labels.size(0)
    return correct / total


def train(epochs=12, n_points=512, batch_size=32, lr=1e-3, reg_weight=1e-3,
          use_tnet=True, device=None, save_path=None, verbose=True, seed=0):
    """Train the classifier on the synthetic dataset and return (model, acc)."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)

    train_set = ShapeDataset(n_samples=1200, n_points=n_points, seed=seed)
    test_set = ShapeDataset(n_samples=300, n_points=n_points, seed=seed + 999)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size)

    model = PointNetClassifier(len(CLASSES), use_tnet=use_tnet).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=5, gamma=0.5)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        for pts, labels in train_loader:
            pts, labels = pts.to(device), labels.to(device)
            logits, feat_mat, _ = model(pts)
            loss = criterion(logits, labels)
            loss = loss + reg_weight * feature_transform_regularizer(feat_mat)
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
        if verbose:
            acc = evaluate(model, test_loader, device)
            print(f"epoch {epoch + 1:2d}/{epochs}  loss {loss.item():.3f}  test_acc {acc:.3f}")

    acc = evaluate(model, test_loader, device)
    if save_path:
        torch.save(model.state_dict(), save_path)
    return model, acc


if __name__ == "__main__":
    _, acc = train(save_path="pointnet.pt")
    print(f"final test accuracy: {acc:.3f}")
