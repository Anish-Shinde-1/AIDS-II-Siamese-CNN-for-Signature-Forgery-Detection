"""
Convenience entrypoint for signature verification.
Forwards to src.verify.main().
"""
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.verify import main, verify_signatures, load_verification_model, preprocess_signature

if __name__ == "__main__":
    main()
