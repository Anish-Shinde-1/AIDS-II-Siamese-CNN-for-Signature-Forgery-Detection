"""
Convenience entrypoint for training the Siamese signature verification model.
Forwards to src.train.
"""
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.train import train, ContrastiveLoss

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train Siamese Signature Verification Model")
    parser.add_argument("--data_dir", type=str, default="data/signatures")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--margin", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--pairs_per_writer", type=int, default=30)
    parser.add_argument("--save_path", type=str, default="checkpoints/best_siamese_model.pth")
    parser.add_argument("--device", type=str, default="cpu")

    args = parser.parse_args()
    train(
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        margin=args.margin,
        patience=args.patience,
        pairs_per_writer=args.pairs_per_writer,
        save_path=args.save_path,
        device_name=args.device,
    )
