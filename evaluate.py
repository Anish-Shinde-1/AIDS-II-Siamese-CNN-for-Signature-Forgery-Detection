"""
Convenience entrypoint for evaluating the Siamese signature verification model.
Forwards to src.evaluate.
"""
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.evaluate import evaluate, compute_test_distances, find_optimal_threshold_and_eer

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate Siamese Signature Verification Model")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_siamese_model.pth")
    parser.add_argument("--data_dir", type=str, default="data/signatures")
    parser.add_argument("--pairs_per_writer", type=int, default=120)
    parser.add_argument("--output_dir", type=str, default="evaluation_output")
    parser.add_argument("--device", type=str, default="cpu")

    args = parser.parse_args()
    evaluate(
        checkpoint_path=args.checkpoint,
        data_dir=args.data_dir,
        pairs_per_writer=args.pairs_per_writer,
        output_dir=args.output_dir,
        device_name=args.device,
    )
