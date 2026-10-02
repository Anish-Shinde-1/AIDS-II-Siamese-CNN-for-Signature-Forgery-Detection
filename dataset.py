import os
import random
from itertools import combinations
from pathlib import Path
from typing import List, Tuple, Optional

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


class SignaturePairDataset(Dataset):
    """
    PyTorch Dataset for offline handwritten signature verification on CEDAR dataset.
    Generates balanced positive (genuine-genuine) and negative (genuine-forged) pairs
    under a strict writer-independent regime.
    """

    def __init__(
        self,
        data_dir: str = "data/signatures",
        writers: Optional[List[int]] = None,
        img_size: Tuple[int, int] = (155, 220),  # (height, width)
        pairs_per_writer: Optional[int] = 120,   # Number of positive & negative pairs per writer
        seed: int = 42,
        augment: bool = False,
    ):
        """
        Args:
            data_dir: Path to directory containing 'full_org' and 'full_forg'
            writers: List of writer IDs (1 to 55) to include in this split
            img_size: Target image size (H, W)
            pairs_per_writer: Number of positive and negative pairs per writer (None for all 276)
            seed: Random seed for reproducibility
            augment: Whether to apply data augmentation for training
        """
        self.data_dir = Path(data_dir)
        self.org_dir = self.data_dir / "full_org"
        self.forg_dir = self.data_dir / "full_forg"
        self.img_size = img_size
        self.writers = writers if writers is not None else list(range(1, 56))

        # Transformations
        transform_list = [
            transforms.Resize(self.img_size),
            transforms.Grayscale(num_output_channels=1),
        ]
        if augment:
            transform_list.extend([
                transforms.RandomRotation(degrees=(-5, 5), fill=255),
                transforms.RandomAffine(degrees=0, translate=(0.02, 0.02), fill=255),
            ])
        transform_list.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),  # Scales pixels to [-1, 1]
        ])
        self.transform = transforms.Compose(transform_list)

        # Generate deterministic pairs
        self.pairs: List[Tuple[str, str, float]] = []
        self._generate_pairs(pairs_per_writer, seed)

    def _generate_pairs(self, pairs_per_writer: Optional[int], seed: int):
        rng = random.Random(seed)

        for writer in self.writers:
            # Collect 24 originals and 24 forgeries for this writer
            org_files = [
                str(self.org_dir / f"original_{writer}_{s}.png")
                for s in range(1, 25)
                if (self.org_dir / f"original_{writer}_{s}.png").exists()
            ]
            forg_files = [
                str(self.forg_dir / f"forgeries_{writer}_{s}.png")
                for s in range(1, 25)
                if (self.forg_dir / f"forgeries_{writer}_{s}.png").exists()
            ]

            if len(org_files) < 2 or len(forg_files) < 1:
                continue

            # 1. Positive Pairs (Genuine vs Genuine) -> Label 0.0
            pos_pairs = list(combinations(org_files, 2))
            if pairs_per_writer is not None and pairs_per_writer < len(pos_pairs):
                pos_pairs = rng.sample(pos_pairs, pairs_per_writer)
            for f1, f2 in pos_pairs:
                self.pairs.append((f1, f2, 0.0))

            # 2. Negative Pairs (Genuine vs Skilled Forgery) -> Label 1.0
            neg_pool = [(f_org, f_forg) for f_org in org_files for f_forg in forg_files]
            num_neg = len(pos_pairs)  # Guarantee 1:1 balance
            neg_pairs = rng.sample(neg_pool, min(num_neg, len(neg_pool)))
            for f1, f2 in neg_pairs:
                self.pairs.append((f1, f2, 1.0))

        # Shuffle all pairs across writers
        rng.shuffle(self.pairs)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        f1, f2, label = self.pairs[idx]

        img1 = Image.open(f1).convert("L")
        img2 = Image.open(f2).convert("L")

        img1 = self.transform(img1)
        img2 = self.transform(img2)
        label_tensor = torch.tensor(label, dtype=torch.float32)

        return img1, img2, label_tensor


def get_dataloaders(
    data_dir: str = "data/signatures",
    batch_size: int = 32,
    num_workers: int = 0,
    pairs_per_writer: int = 120,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Constructs train, validation, and test DataLoaders under strict writer independence:
      - Training: Writers 1 to 35
      - Validation: Writers 36 to 40
      - Testing: Writers 41 to 55 (Unseen evaluation writers)
    """
    train_writers = list(range(1, 36))
    val_writers = list(range(36, 41))
    test_writers = list(range(41, 56))

    train_dataset = SignaturePairDataset(
        data_dir=data_dir,
        writers=train_writers,
        pairs_per_writer=pairs_per_writer,
        seed=seed,
        augment=True,
    )

    val_dataset = SignaturePairDataset(
        data_dir=data_dir,
        writers=val_writers,
        pairs_per_writer=pairs_per_writer,
        seed=seed,
        augment=False,
    )

    test_dataset = SignaturePairDataset(
        data_dir=data_dir,
        writers=test_writers,
        pairs_per_writer=pairs_per_writer,
        seed=seed,
        augment=False,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    train_loader, val_loader, test_loader = get_dataloaders(pairs_per_writer=20)
    print(f"Train batches: {len(train_loader)} (Samples: {len(train_loader.dataset)})")
    print(f"Val batches:   {len(val_loader)} (Samples: {len(val_loader.dataset)})")
    print(f"Test batches:  {len(test_loader)} (Samples: {len(test_loader.dataset)})")

    img1, img2, label = next(iter(train_loader))
    print(f"img1 shape: {img1.shape}, img2 shape: {img2.shape}, label shape: {label.shape}")
    print(f"Sample labels (0=Genuine, 1=Forged): {label[:8].tolist()}")
