import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

from dataset import get_dataloaders
from model import SiameseNetwork


class ContrastiveLoss(nn.Module):
    """
    Contrastive Loss Function for Metric Learning:
      L(y, d) = 0.5 * (1 - y) * d^2 + 0.5 * y * max(0, margin - d)^2

    Label Conventions:
      y = 0: Genuine Pair (Positive) -> Loss penalizes distance d > 0
      y = 1: Forged Pair  (Negative) -> Loss penalizes distance d < margin
    """

    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = margin

    def forward(self, distance: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss_genuine = (1.0 - target) * torch.pow(distance, 2)
        loss_forged = target * torch.pow(torch.clamp(self.margin - distance, min=0.0), 2)
        loss = 0.5 * (loss_genuine + loss_forged)
        return torch.mean(loss)


def train_epoch(
    model: nn.Module,
    dataloader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for x1, x2, labels in dataloader:
        x1, x2, labels = x1.to(device), x2.to(device), labels.to(device)

        optimizer.zero_grad()
        distances, _, _ = model(x1, x2)
        loss = criterion(distances, labels)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        optimizer.step()

        running_loss += loss.item() * x1.size(0)

        # Interim accuracy evaluated at margin/2 (0.5)
        threshold = criterion.margin / 2.0
        preds = (distances > threshold).float()
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def validate_epoch(
    model: nn.Module,
    dataloader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for x1, x2, labels in dataloader:
            x1, x2, labels = x1.to(device), x2.to(device), labels.to(device)

            distances, _, _ = model(x1, x2)
            loss = criterion(distances, labels)

            running_loss += loss.item() * x1.size(0)
            threshold = criterion.margin / 2.0
            preds = (distances > threshold).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def train(
    data_dir: str = "data/signatures",
    epochs: int = 10,
    batch_size: int = 32,
    lr: float = 1e-3,
    margin: float = 1.0,
    patience: int = 4,
    pairs_per_writer: int = 30,
    save_path: str = "best_siamese_model.pth",
    device_name: str = "cpu",
):
    if device_name == "cpu":
        torch.set_num_threads(4)

    device = torch.device(device_name if torch.cuda.is_available() or device_name == "cpu" else "cpu")
    print(f"[*] Training on device: {device}", flush=True)

    print(f"[*] Initializing dataset from: {data_dir}", flush=True)
    train_loader, val_loader, _ = get_dataloaders(
        data_dir=data_dir,
        batch_size=batch_size,
        pairs_per_writer=pairs_per_writer,
    )
    print(
        f"[*] Training pairs: {len(train_loader.dataset)} | Validation pairs: {len(val_loader.dataset)}",
        flush=True,
    )

    model = SiameseNetwork(in_channels=1, embedding_dim=128).to(device)
    criterion = ContrastiveLoss(margin=margin)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    best_val_loss = float("inf")
    patience_counter = 0
    history: Dict[str, List[float]] = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "lr": [],
    }

    start_time = time.time()
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        current_lr = optimizer.param_groups[0]["lr"]
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["lr"].append(current_lr)

        elapsed = time.time() - t0
        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] ({elapsed:.1f}s) "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}% | LR: {current_lr:.1e}",
            flush=True,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "val_acc": val_acc,
                "margin": margin,
            }
            torch.save(checkpoint, save_path)
            print(f"  --> Saved new best checkpoint to {save_path} (Val Loss: {val_loss:.4f})", flush=True)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"[*] Early stopping triggered after {epoch} epochs (Patience: {patience}).", flush=True)
                break

    total_time = time.time() - start_time
    print(f"[*] Training completed in {total_time:.1f}s. Best Val Loss: {best_val_loss:.4f}", flush=True)

    with open("training_history.json", "w") as f:
        json.dump(history, f, indent=2)
    print("[*] Saved training history to training_history.json", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Siamese Signature Verification Model")
    parser.add_argument("--data_dir", type=str, default="data/signatures")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--margin", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--pairs_per_writer", type=int, default=30)
    parser.add_argument("--save_path", type=str, default="best_siamese_model.pth")
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
