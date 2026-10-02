import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Tuple

import torch
from torchvision import transforms
from PIL import Image

# Ensure src and project root are in sys.path
SRC_DIR = Path(__file__).resolve().parent
ROOT_DIR = SRC_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

try:
    from model import SiameseNetwork
except ImportError:
    from src.model import SiameseNetwork


def resolve_checkpoint_path(path_str: str) -> Path:
    p = Path(path_str)
    if p.exists():
        return p
    candidates = [
        ROOT_DIR / path_str,
        ROOT_DIR / "checkpoints" / path_str,
        ROOT_DIR / "checkpoints" / p.name,
        SRC_DIR / path_str,
        ROOT_DIR / "best_siamese_model.pth",
    ]
    for cand in candidates:
        if cand.exists():
            return cand
    return p


def load_verification_model(
    checkpoint_path: str = "checkpoints/best_siamese_model.pth",
    device: Optional[torch.device] = None,
) -> Tuple[SiameseNetwork, float]:
    """
    Loads trained SiameseNetwork and determines decision threshold from
    checkpoint/evaluation metadata.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_path = resolve_checkpoint_path(checkpoint_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Checkpoint '{checkpoint_path}' not found (checked {ckpt_path}). "
            "Please verify model path or train the model using train.py."
        )

    model = SiameseNetwork(in_channels=1, embedding_dim=128).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)

    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)

    model.eval()

    # Look for saved threshold in evaluation results
    threshold = 0.5332  # Default empirically optimal threshold
    eval_candidates = [
        Path("evaluation_output/evaluation_results.json"),
        ROOT_DIR / "evaluation_output" / "evaluation_results.json",
        ROOT_DIR / "checkpoints" / "evaluation_results.json",
    ]
    for eval_json in eval_candidates:
        if eval_json.exists():
            try:
                with open(eval_json, "r") as f:
                    data = json.load(f)
                    threshold = float(data.get("optimal_threshold", threshold))
                break
            except Exception:
                pass

    return model, threshold


def preprocess_signature(image_path: str, img_size: Tuple[int, int] = (155, 220)) -> torch.Tensor:
    """Preprocesses a single signature image identically to the training pipeline."""
    p = Path(image_path)
    if not p.exists() and (ROOT_DIR / image_path).exists():
        p = ROOT_DIR / image_path

    if not p.exists():
        raise FileNotFoundError(f"Signature image not found at: {image_path}")

    img = Image.open(p).convert("L")
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])
    tensor = transform(img).unsqueeze(0)  # Shape: (1, 1, 155, 220)
    return tensor


def verify_signatures(
    img1_path: str,
    img2_path: str,
    checkpoint_path: str = "checkpoints/best_siamese_model.pth",
    threshold: Optional[float] = None,
    device_name: str = "cpu",
) -> Tuple[bool, float, float]:
    """
    Verifies whether two signature images belong to the same genuine writer.
    Returns:
        is_match: True if genuine match, False if forgery detected
        distance: Euclidean distance between embeddings
        similarity_pct: Percentage similarity score (0% to 100%)
    """
    device = torch.device(device_name if torch.cuda.is_available() or device_name == "cpu" else "cpu")
    model, default_threshold = load_verification_model(checkpoint_path, device)
    tau = threshold if threshold is not None else default_threshold

    t1 = preprocess_signature(img1_path).to(device)
    t2 = preprocess_signature(img2_path).to(device)

    with torch.no_grad():
        dist, _, _ = model(t1, t2)
        distance = float(dist.item())

    # Decision rule: Distance <= threshold indicates genuine match
    is_match = distance <= tau

    # Calculate intuitive similarity percentage
    # When distance is 0, similarity is 100%; when distance reaches 2.0 (opposite unit vectors), similarity is 0%.
    similarity_pct = max(0.0, min(100.0, (1.0 - (distance / 2.0)) * 100.0))

    return is_match, distance, similarity_pct


def main():
    parser = argparse.ArgumentParser(
        description="Verify authenticity between two handwritten signature images"
    )
    parser.add_argument("--img1", type=str, required=True, help="Path to reference / original signature")
    parser.add_argument("--img2", type=str, required=True, help="Path to query / test signature")
    parser.add_argument(
        "--checkpoint", type=str, default="checkpoints/best_siamese_model.pth", help="Model checkpoint path"
    )
    parser.add_argument(
        "--threshold", type=float, default=None, help="Decision distance threshold (optional)"
    )
    parser.add_argument("--device", type=str, default="cpu")

    args = parser.parse_args()

    is_match, distance, similarity = verify_signatures(
        img1_path=args.img1,
        img2_path=args.img2,
        checkpoint_path=args.checkpoint,
        threshold=args.threshold,
        device_name=args.device,
    )

    threshold_val = args.threshold if args.threshold is not None else "Auto (optimal EER/Acc)"

    print("\n" + "=" * 50)
    print("      OFFLINE SIGNATURE VERIFICATION RESULT       ")
    print("=" * 50)
    print(f" Reference Image:  {args.img1}")
    print(f" Query Image:      {args.img2}")
    print(f" Euclidean Dist:   {distance:.4f}")
    print(f" Active Threshold: {threshold_val}")
    print(f" Similarity Score: {similarity:.2f}%")
    print("-" * 50)
    if is_match:
        print(" VERDICT:          [+] GENUINE MATCH")
        print(" Detail:           The signatures are statistically verified to be authentic.")
    else:
        print(" VERDICT:          [-] FORGERY DETECTED")
        print(" Detail:           Discrepancy exceeds threshold; signature flagged as a forgery.")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    main()
