"""
Train a ternary credit risk classifier using the Triton framework.

Uses Triton's TernaryLinear layers, STE gradient flow, and periodic
weight requantization for a production-grade ternary text classifier.

Usage:
    # Generate training data first (from Ternary-SD or copy data/)
    python models/scripts/train_credit_risk.py \
        --data data/risk_training.jsonl \
        --epochs 50 \
        --batch_size 32

    # With early stopping and label smoothing
    python models/scripts/train_credit_risk.py \
        --data data/risk_training_v2.jsonl \
        --epochs 100 \
        --early_stopping --early_stopping_patience 15 \
        --label_smoothing 0.1

    # Resume from checkpoint
    python models/scripts/train_credit_risk.py \
        --data data/risk_training_v2.jsonl \
        --resume checkpoints/credit_risk_best.pth
"""

import argparse
import csv
import json
import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from models.credit_risk.ternary_credit_risk import (
    TernaryCreditRiskNet,
    CreditRiskTokenizer,
    RISK_LABELS,
    quantize_model_weights,
)

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False


# ── Dataset ─────────────────────────────────────────────────────────

class CreditRiskDataset(Dataset):
    """JSONL dataset of (borrower notes -> risk label) pairs."""

    LABEL_MAP = {"Low": 0, "Medium": 1, "High": 2}

    def __init__(self, path: str, tokenizer: CreditRiskTokenizer):
        self.examples = []
        self.tokenizer = tokenizer

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line.strip())
                text = obj["input"]
                label = self.LABEL_MAP.get(obj["output"]["Risk_Flag"], 1)
                self.examples.append((text, label))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        text, label = self.examples[idx]
        input_ids = self.tokenizer.encode(text)
        return input_ids, torch.tensor(label, dtype=torch.long)


# ── Label smoothing (reused from Triton training infra) ─────────────

class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, pred, target):
        n_classes = pred.size(-1)
        log_probs = nn.functional.log_softmax(pred, dim=-1)
        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)
            true_dist.fill_(self.smoothing / (n_classes - 1))
            true_dist.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing)
        return torch.mean(torch.sum(-true_dist * log_probs, dim=-1))


# ── Early stopping (reused from Triton) ─────────────────────────────

class EarlyStopping:
    def __init__(self, patience=15, min_delta=0.0, mode="max"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.should_stop = False

    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
            return False
        improved = (
            score > self.best_score + self.min_delta
            if self.mode == "max"
            else score < self.best_score - self.min_delta
        )
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
        if self.counter >= self.patience:
            self.should_stop = True
        return self.should_stop


# ── Training ────────────────────────────────────────────────────────

def train_epoch(model, loader, criterion, optimizer, device, epoch, log_interval=50):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch_idx, (input_ids, labels) in enumerate(loader):
        input_ids, labels = input_ids.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(input_ids)
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # Periodic ternary requantization
        if batch_idx % 50 == 0:
            quantize_model_weights(model)

        total_loss += loss.item()
        _, predicted = logits.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        if (batch_idx + 1) % log_interval == 0:
            print(
                f"  Epoch {epoch} | Batch {batch_idx + 1}/{len(loader)} | "
                f"Loss: {total_loss / (batch_idx + 1):.4f} | "
                f"Acc: {100. * correct / total:.1f}%"
            )

    return total_loss / len(loader), 100.0 * correct / total


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    per_class_correct = [0, 0, 0]
    per_class_total = [0, 0, 0]

    with torch.no_grad():
        for input_ids, labels in loader:
            input_ids, labels = input_ids.to(device), labels.to(device)
            logits = model(input_ids)
            loss = criterion(logits, labels)

            total_loss += loss.item()
            _, predicted = logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            for cls in range(3):
                mask = labels == cls
                per_class_total[cls] += mask.sum().item()
                per_class_correct[cls] += (predicted[mask] == cls).sum().item()

    val_loss = total_loss / len(loader)
    val_acc = 100.0 * correct / total

    per_class_acc = {}
    for cls in range(3):
        if per_class_total[cls] > 0:
            per_class_acc[RISK_LABELS[cls]] = round(
                100.0 * per_class_correct[cls] / per_class_total[cls], 1
            )
        else:
            per_class_acc[RISK_LABELS[cls]] = 0.0

    return val_loss, val_acc, per_class_acc


# ── Main ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train Ternary Credit Risk Classifier")

    # Data
    parser.add_argument("--data", required=True, help="JSONL training data")
    parser.add_argument("--val_split", type=float, default=0.1)

    # Model
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--max_length", type=int, default=128)

    # Training
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--scheduler", choices=["cosine", "step", "none"], default="cosine")
    parser.add_argument("--label_smoothing", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)

    # Early stopping
    parser.add_argument("--early_stopping", action="store_true")
    parser.add_argument("--early_stopping_patience", type=int, default=15)

    # Checkpointing
    parser.add_argument("--checkpoint_dir", default="./checkpoints")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--save_freq", type=int, default=10)

    # Logging
    parser.add_argument("--log_dir", default="./logs")

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    # ── Load data + build tokenizer ──
    print(f"Loading data from {args.data}...")
    all_texts = []
    with open(args.data, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line.strip())
            all_texts.append(obj["input"])

    tokenizer = CreditRiskTokenizer(max_length=args.max_length)
    tokenizer.build_vocab(all_texts)
    print(f"Vocabulary: {tokenizer.vocab_size} tokens")

    dataset = CreditRiskDataset(args.data, tokenizer)
    val_size = int(len(dataset) * args.val_split)
    train_size = len(dataset) - val_size
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, num_workers=0)

    print(f"Train: {train_size}, Val: {val_size}")

    # ── Create model ──
    model = TernaryCreditRiskNet(
        vocab_size=tokenizer.vocab_size,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    ternary_params = sum(
        p.numel() for n, p in model.named_parameters() if "fc" in n and "weight" in n
    )
    print(f"Parameters: {num_params:,} total, {ternary_params:,} ternary")
    print(f"Estimated ternary size: {ternary_params * 2.5 / 8 / 1024:.1f} KB")

    # ── Loss, optimizer, scheduler ──
    if args.label_smoothing > 0:
        criterion = LabelSmoothingCrossEntropy(args.label_smoothing)
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    scheduler = None
    if args.scheduler == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    elif args.scheduler == "step":
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

    early_stopping = None
    if args.early_stopping:
        early_stopping = EarlyStopping(patience=args.early_stopping_patience)

    # TensorBoard
    writer = None
    if TENSORBOARD_AVAILABLE:
        writer = SummaryWriter(os.path.join(args.log_dir, f"credit_risk_{time.strftime('%Y%m%d_%H%M%S')}"))

    # CSV log
    csv_path = os.path.join(args.log_dir, "credit_risk_metrics.csv")
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc",
                         "low_acc", "med_acc", "high_acc", "lr", "best_acc"])

    # Resume
    start_epoch = 0
    best_acc = 0.0
    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if scheduler and ckpt.get("scheduler_state_dict"):
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_acc = ckpt.get("accuracy", 0.0)
        print(f"Resumed from epoch {start_epoch}, best acc: {best_acc:.1f}%")

    # ── Training loop ──
    print(f"\n{'='*70}")
    print(f"Training Ternary Credit Risk Classifier")
    print(f"  Model: TernaryCreditRiskNet (embed={args.embed_dim}, hidden={args.hidden_dim})")
    print(f"  Epochs: {start_epoch} -> {args.epochs}")
    print(f"  Batch: {args.batch_size}, LR: {args.lr}, Scheduler: {args.scheduler}")
    print(f"{'='*70}\n")

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch + 1
        )
        val_loss, val_acc, per_class = evaluate(model, val_loader, criterion, device)
        elapsed = time.time() - t0

        lr = optimizer.param_groups[0]["lr"]
        if scheduler:
            scheduler.step()

        is_best = val_acc > best_acc
        if is_best:
            best_acc = val_acc

        print(f"\nEpoch {epoch + 1}/{args.epochs}:")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.1f}%")
        print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.1f}%")
        print(f"  Per-class:  Low={per_class['Low']:.1f}%  Med={per_class['Medium']:.1f}%  High={per_class['High']:.1f}%")
        print(f"  LR: {lr:.6f} | Time: {elapsed:.1f}s")
        if is_best:
            print(f"  *** New best: {best_acc:.1f}% ***")

        # Log
        if writer:
            writer.add_scalar("train/loss", train_loss, epoch)
            writer.add_scalar("train/acc", train_acc, epoch)
            writer.add_scalar("val/loss", val_loss, epoch)
            writer.add_scalar("val/acc", val_acc, epoch)

        csv_writer.writerow([
            epoch + 1, train_loss, train_acc, val_loss, val_acc,
            per_class["Low"], per_class["Medium"], per_class["High"],
            lr, best_acc,
        ])
        csv_file.flush()

        # Checkpoints
        if (epoch + 1) % args.save_freq == 0:
            path = os.path.join(args.checkpoint_dir, f"credit_risk_epoch_{epoch + 1}.pth")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
                "accuracy": val_acc,
                "vocab_size": tokenizer.vocab_size,
            }, path)

        if is_best:
            best_path = os.path.join(args.checkpoint_dir, "credit_risk_best.pth")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
                "accuracy": val_acc,
                "vocab_size": tokenizer.vocab_size,
            }, best_path)

        # Save tokenizer alongside model
        tokenizer.save(os.path.join(args.checkpoint_dir, "tokenizer.json"))

        if early_stopping and early_stopping(val_acc):
            print(f"\nEarly stopping at epoch {epoch + 1} (best: {best_acc:.1f}%)")
            break

    csv_file.close()
    if writer:
        writer.close()

    # ── Final stats ──
    print(f"\n{'='*70}")
    print(f"Training complete! Best accuracy: {best_acc:.1f}%")
    print(f"Model saved to: {args.checkpoint_dir}/credit_risk_best.pth")
    print(f"Tokenizer saved to: {args.checkpoint_dir}/tokenizer.json")
    print(f"Ternary model size: ~{ternary_params * 2.5 / 8 / 1024:.1f} KB")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
