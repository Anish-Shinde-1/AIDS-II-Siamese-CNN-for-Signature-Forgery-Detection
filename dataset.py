"""
Convenience module for dataset utilities.
Forwards to src.dataset.
"""
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.dataset import SignaturePairDataset, get_dataloaders, resolve_data_dir

if __name__ == "__main__":
    train_loader, val_loader, test_loader = get_dataloaders(pairs_per_writer=20)
    print(f"Train batches: {len(train_loader)} (Samples: {len(train_loader.dataset)})")
    print(f"Val batches:   {len(val_loader)} (Samples: {len(val_loader.dataset)})")
    print(f"Test batches:  {len(test_loader)} (Samples: {len(test_loader.dataset)})")
