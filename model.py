import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class SignatureFeatureExtractor(nn.Module):
    """
    Lightweight, high-performance Convolutional Backbone for offline signature verification.
    Replaces the bloated 445 MB Dense(1024) layer from the original notebook with
    Global Average Pooling (AdaptiveAvgPool2d), keeping parameter count < 1.7 MB (< 25 MB footprint)
    while effectively preventing spatial overfitting and speeding up inference.
    """

    def __init__(self, in_channels: int = 1, embedding_dim: int = 128):
        super().__init__()
        self.embedding_dim = embedding_dim

        self.features = nn.Sequential(
            # Block 1: (B, 1, 155, 220) -> (B, 32, 78, 110)
            nn.Conv2d(in_channels, 32, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            # Block 2: (B, 32, 78, 110) -> (B, 64, 39, 55)
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1),

            # Block 3: (B, 64, 39, 55) -> (B, 128, 20, 28)
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.15),

            # Block 4: (B, 128, 20, 28) -> (B, 256, 10, 14)
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            # Global Average Pooling replaces flattened dense bloat
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        # Projection Head into L2-normalized embedding space
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128, bias=False),
            nn.BatchNorm1d(128),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.features(x)
        emb = self.head(feat)
        # Unit hypersphere projection
        emb = F.normalize(emb, p=2, dim=-1)
        return emb


class SiameseNetwork(nn.Module):
    """
    Twin weight-sharing Siamese Architecture for pairwise distance metric learning.
    """

    def __init__(self, in_channels: int = 1, embedding_dim: int = 128):
        super().__init__()
        self.backbone = SignatureFeatureExtractor(
            in_channels=in_channels, embedding_dim=embedding_dim
        )

    def forward_one(self, x: torch.Tensor) -> torch.Tensor:
        """Extract L2-normalized embedding for a single signature."""
        return self.backbone(x)

    def forward(
        self, x1: torch.Tensor, x2: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Pass pair through shared backbone and calculate pairwise Euclidean distance.
        Returns:
            distance: Tensor of shape (B,) with Euclidean distance between embeddings
            emb1: Tensor of shape (B, embedding_dim)
            emb2: Tensor of shape (B, embedding_dim)
        """
        emb1 = self.forward_one(x1)
        emb2 = self.forward_one(x2)
        # Numerical stability epsilon added before square root
        distance = F.pairwise_distance(emb1, emb2, p=2.0, eps=1e-7)
        return distance, emb1, emb2


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    net = SiameseNetwork(in_channels=1, embedding_dim=128)
    params = count_parameters(net)
    print(f"Total Trainable Parameters: {params:,}")
    print(f"Estimated Checkpoint Size: {params * 4 / (1024 * 1024):.2f} MB")

    # Smoke test forward pass
    dummy_x1 = torch.randn(8, 1, 155, 220)
    dummy_x2 = torch.randn(8, 1, 155, 220)
    dist, e1, e2 = net(dummy_x1, dummy_x2)

    print(f"Embedding shape: {e1.shape}")
    print(f"L2 Norm check (should be ~1.0): {torch.norm(e1[0]).item():.4f}")
    print(f"Distance tensor shape: {dist.shape}")
    print(f"Sample pairwise distances: {dist[:4].tolist()}")
