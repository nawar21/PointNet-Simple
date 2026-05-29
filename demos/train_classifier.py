"""Train the PointNet classifier on the synthetic dataset.

    python -m demos.train_classifier
"""
from pointnet import train

if __name__ == "__main__":
    model, acc = train(epochs=12, n_points=512, save_path="pointnet.pt")
    print(f"\nSaved checkpoint to pointnet.pt  |  final test accuracy: {acc:.3f}")
