"""
Convenience module for model architecture.
Forwards to src.model.
"""
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.model import SignatureFeatureExtractor, SiameseNetwork, count_parameters

if __name__ == "__main__":
    net = SiameseNetwork(in_channels=1, embedding_dim=128)
    params = count_parameters(net)
    print(f"Total Trainable Parameters: {params:,}")
