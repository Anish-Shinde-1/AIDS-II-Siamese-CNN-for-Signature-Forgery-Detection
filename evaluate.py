import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    auc,
)

from dataset import SignaturePairDataset
from model import SiameseNetwork


def compute_test_distances(
    model: torch.nn.Module,
    test_dataset: SignaturePairDataset,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes pairwise Euclidean distances for all test pairs.
    Returns:
        distances: 1D numpy array of Euclidean distances
        labels: 1D numpy array of true labels (0 = Genuine, 1 = Forged)
    """
    model.eval()
    dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=32, shuffle=False, num_workers=0
    )

    all_distances = []
    all_labels = []

    with torch.no_grad():
        for x1, x2, labels in dataloader:
            x1, x2 = x1.to(device), x2.to(device)
            distances, _, _ = model(x1, x2)
            all_distances.extend(distances.cpu().numpy().tolist())
            all_labels.extend(labels.numpy().tolist())

    return np.array(all_distances), np.array(all_labels)


def find_optimal_threshold_and_eer(
    distances: np.ndarray,
    labels: np.ndarray,
    num_thresholds: int = 1000,
) -> Tuple[float, float, float, Dict]:
    """
    Sweeps decision threshold tau:
      If distance <= tau: predict 0 (Genuine)
      If distance > tau:  predict 1 (Forged)

    Computes:
      - FAR (False Acceptance Rate): Forged signatures accepted as Genuine
      - FRR (False Rejection Rate): Genuine signatures rejected as Forged
      - EER (Equal Error Rate): Point where FAR == FRR
      - Optimal Threshold minimizing total error (FAR + FRR)
    """
    genuine_mask = labels == 0
    forged_mask = labels == 1

    total_genuine = np.sum(genuine_mask)
    total_forged = np.sum(forged_mask)

    min_dist, max_dist = np.min(distances), np.max(distances)
    thresholds = np.linspace(min_dist, max_dist, num_thresholds)

    fars = []
    frrs = []
    accuracies = []

    best_threshold = 0.5
    best_acc = 0.0
    eer = 1.0
    eer_threshold = 0.5
    min_diff = float("inf")

    for tau in thresholds:
        # Distance <= tau => predicted Genuine (0)
        pred_genuine = distances <= tau
        # Distance > tau => predicted Forged (1)
        pred_forged = distances > tau

        # FAR: Forged classified as Genuine
        fa = np.sum(forged_mask & pred_genuine)
        far = fa / total_forged if total_forged > 0 else 0.0

        # FRR: Genuine classified as Forged
        fr = np.sum(genuine_mask & pred_forged)
        frr = fr / total_genuine if total_genuine > 0 else 0.0

        # Accuracy
        preds = pred_forged.astype(int)  # 0 for genuine, 1 for forged
        acc = accuracy_score(labels, preds)

        fars.append(far)
        frrs.append(frr)
        accuracies.append(acc)

        if acc > best_acc:
            best_acc = acc
            best_threshold = tau

        diff = abs(far - frr)
        if diff < min_diff:
            min_diff = diff
            eer = (far + frr) / 2.0
            eer_threshold = tau

    sweep_data = {
        "thresholds": thresholds,
        "fars": np.array(fars),
        "frrs": np.array(frrs),
        "accuracies": np.array(accuracies),
    }

    return best_threshold, eer, eer_threshold, sweep_data


def plot_distributions_and_roc(
    distances: np.ndarray,
    labels: np.ndarray,
    best_threshold: float,
    eer: float,
    eer_threshold: float,
    output_dir: Path,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    genuine_dists = distances[labels == 0]
    forged_dists = distances[labels == 1]

    # 1. Distance Distribution Histogram
    plt.figure(figsize=(9, 5))
    bins = np.linspace(0, max(np.max(distances) + 0.1, 1.5), 40)
    plt.hist(
        genuine_dists,
        bins=bins,
        alpha=0.65,
        color="#2ecc71",
        label=f"Genuine Pairs (N={len(genuine_dists)})",
        density=True,
    )
    plt.hist(
        forged_dists,
        bins=bins,
        alpha=0.65,
        color="#e74c3c",
        label=f"Forged Pairs (N={len(forged_dists)})",
        density=True,
    )
    plt.axvline(
        best_threshold,
        color="#2980b9",
        linestyle="--",
        linewidth=2,
        label=f"Best Threshold (τ={best_threshold:.3f})",
    )
    plt.axvline(
        eer_threshold,
        color="#8e44ad",
        linestyle=":",
        linewidth=2,
        label=f"EER Threshold (τ={eer_threshold:.3f})",
    )
    plt.title("Signature Verification: Distance Distribution (Unseen Test Writers)", fontsize=13)
    plt.xlabel("Euclidean Distance between Embeddings", fontsize=11)
    plt.ylabel("Density", fontsize=11)
    plt.legend(frameon=True, facecolor="white", framealpha=0.9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    dist_plot_path = output_dir / "distance_distribution.png"
    plt.savefig(dist_plot_path, dpi=200)
    plt.close()
    print(f"[*] Saved distance distribution plot to: {dist_plot_path}")

    # 2. ROC Curve
    # For ROC, higher score should indicate positive class (Forged = 1).
    # Since higher distance indicates forgery, distances act as forgery scores!
    fpr, tpr, _ = roc_curve(labels, distances, pos_label=1)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, color="#3498db", lw=2.5, label=f"ROC Curve (AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], color="#7f8c8d", linestyle="--", lw=1.5, label="Random Guess (AUC=0.50)")
    plt.scatter([eer], [1.0 - eer], color="#e74c3c", s=70, zorder=5, label=f"EER Point ({eer*100:.2f}%)")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (FAR: Forged accepted as Genuine)", fontsize=11)
    plt.ylabel("True Positive Rate (1 - FRR: Forged correctly detected)", fontsize=11)
    plt.title("ROC Curve - Offline Signature Verification", fontsize=13)
    plt.legend(loc="lower right", frameon=True, facecolor="white", framealpha=0.9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    roc_plot_path = output_dir / "roc_curve.png"
    plt.savefig(roc_plot_path, dpi=200)
    plt.close()
    print(f"[*] Saved ROC curve plot to: {roc_plot_path}")

    return roc_auc


def evaluate(
    checkpoint_path: str = "best_siamese_model.pth",
    data_dir: str = "data/signatures",
    pairs_per_writer: int = 120,
    output_dir: str = "evaluation_output",
    device_name: str = "cpu",
):
    device = torch.device(device_name if torch.cuda.is_available() or device_name == "cpu" else "cpu")
    print(f"[*] Evaluating checkpoint: {checkpoint_path}")

    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    # Load model
    model = SiameseNetwork(in_channels=1, embedding_dim=128).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)
    print("[*] Model weights successfully restored.")

    # Load test dataset (strictly unseen writers 41 to 55)
    test_writers = list(range(41, 56))
    test_dataset = SignaturePairDataset(
        data_dir=data_dir,
        writers=test_writers,
        pairs_per_writer=pairs_per_writer,
        seed=42,
        augment=False,
    )
    print(f"[*] Test Dataset: {len(test_dataset)} pairs across {len(test_writers)} unseen writers ({test_writers[0]} to {test_writers[-1]})")

    # Inference
    distances, labels = compute_test_distances(model, test_dataset, device)

    # Threshold optimization & EER
    best_tau, eer, eer_tau, _ = find_optimal_threshold_and_eer(distances, labels)

    # Binary metrics at best threshold
    preds = (distances > best_tau).astype(int)  # 0 for genuine, 1 for forged
    acc = accuracy_score(labels, preds)
    prec = precision_score(labels, preds, zero_division=0)
    rec = recall_score(labels, preds, zero_division=0)
    f1 = f1_score(labels, preds, zero_division=0)

    # Metrics at EER threshold
    eer_preds = (distances > eer_tau).astype(int)
    eer_acc = accuracy_score(labels, eer_preds)

    # Plotting & AUC
    out_path = Path(output_dir)
    roc_auc = plot_distributions_and_roc(distances, labels, best_tau, eer, eer_tau, out_path)

    # Calculate final FAR and FRR at best_tau
    genuine_mask = labels == 0
    forged_mask = labels == 1
    far = np.sum(forged_mask & (distances <= best_tau)) / np.sum(forged_mask)
    frr = np.sum(genuine_mask & (distances > best_tau)) / np.sum(genuine_mask)

    results = {
        "test_writers": test_writers,
        "total_test_pairs": len(labels),
        "genuine_pairs": int(np.sum(genuine_mask)),
        "forged_pairs": int(np.sum(forged_mask)),
        "optimal_threshold": float(best_tau),
        "optimal_accuracy": float(acc),
        "optimal_precision": float(prec),
        "optimal_recall": float(rec),
        "optimal_f1": float(f1),
        "optimal_far": float(far),
        "optimal_frr": float(frr),
        "eer": float(eer),
        "eer_threshold": float(eer_tau),
        "eer_accuracy": float(eer_acc),
        "roc_auc": float(roc_auc),
    }

    results_json = out_path / "evaluation_results.json"
    with open(results_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[*] Saved evaluation metrics JSON to: {results_json}")

    print("\n" + "=" * 55)
    print("       OFFLINE SIGNATURE VERIFICATION EVALUATION       ")
    print("=" * 55)
    print(f" Evaluated Writers (Unseen):  {test_writers[0]} to {test_writers[-1]} ({len(test_writers)} individuals)")
    print(f" Total Evaluation Pairs:      {len(labels)} (1:1 balanced)")
    print(f" Area Under ROC Curve (AUC):  {roc_auc:.4f}")
    print(f" Equal Error Rate (EER):      {eer*100:.2f}% (at threshold tau = {eer_tau:.4f})")
    print("-" * 55)
    print(f" Optimal Decision Threshold:  tau = {best_tau:.4f}")
    print(f" Verification Accuracy:       {acc*100:.2f}%")
    print(f" Precision (Forgery):         {prec*100:.2f}%")
    print(f" Recall (Forgery Detection):  {rec*100:.2f}%")
    print(f" F1-Score:                    {f1*100:.2f}%")
    print(f" False Acceptance Rate (FAR): {far*100:.2f}% (Forged passed as genuine)")
    print(f" False Rejection Rate (FRR):  {frr*100:.2f}% (Genuine flagged as forged)")
    print("=" * 55 + "\n")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Siamese Signature Verification Model")
    parser.add_argument("--checkpoint", type=str, default="best_siamese_model.pth")
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
