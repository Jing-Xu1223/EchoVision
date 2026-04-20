#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

# Ensure "src" imports work when executing this file directly.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.musiccaps_dataset import MusicCapsMelDataset
from src.models.music_label_cnn import MusicLabelCNN


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MusicCaps mel->label CNN model.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed/musiccaps"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/music_label_cnn"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--sweep-min-threshold", type=float, default=0.1)
    parser.add_argument("--sweep-max-threshold", type=float, default=0.9)
    parser.add_argument("--sweep-step", type=float, default=0.05)
    parser.add_argument("--target-frames", type=int, default=320)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def micro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tp = float(((y_true == 1) & (y_pred == 1)).sum())
    fp = float(((y_true == 0) & (y_pred == 1)).sum())
    fn = float(((y_true == 1) & (y_pred == 0)).sum())
    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    return 2.0 * precision * recall / (precision + recall + 1e-9)


def threshold_sweep(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    min_threshold: float,
    max_threshold: float,
    step: float,
) -> tuple[float, float]:
    best_threshold = min_threshold
    best_f1 = -1.0
    threshold = min_threshold
    while threshold <= max_threshold + 1e-9:
        y_pred = (y_prob >= threshold).astype(np.float32)
        f1 = micro_f1(y_true, y_pred)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
        threshold += step
    return best_threshold, best_f1


def compute_pos_weight(train_ds: MusicCapsMelDataset) -> torch.Tensor:
    train_indices = np.array([record.index for record in train_ds.records], dtype=np.int64)
    y_train = train_ds.labels[train_indices]
    pos = y_train.sum(axis=0)
    neg = y_train.shape[0] - pos
    pos_weight = (neg + 1.0) / (pos + 1.0)
    return torch.from_numpy(pos_weight.astype(np.float32))


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    threshold: float,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    all_targets: list[np.ndarray] = []
    all_probs: list[np.ndarray] = []

    for mel, target, _ in loader:
        mel = mel.to(device)
        target = target.to(device)

        if is_train:
            optimizer.zero_grad()

        logits = model(mel)
        loss = criterion(logits, target)

        if is_train:
            loss.backward()
            optimizer.step()

        total_loss += float(loss.item()) * mel.shape[0]
        probs = torch.sigmoid(logits)
        all_targets.append(target.detach().cpu().numpy())
        all_probs.append(probs.detach().cpu().numpy())

    y_true = np.concatenate(all_targets, axis=0)
    y_prob = np.concatenate(all_probs, axis=0)
    y_pred = (y_prob >= threshold).astype(np.float32)
    avg_loss = total_loss / len(loader.dataset)
    f1 = micro_f1(y_true, y_pred)
    return avg_loss, f1, y_true, y_prob


def main() -> int:
    args = parse_args()
    seed_everything(args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata_csv = args.data_dir / "metadata.csv"
    labels_npy = args.data_dir / "labels_multihot.npy"
    vocab_path = args.data_dir / "label_vocab.json"

    if not metadata_csv.exists() or not labels_npy.exists() or not vocab_path.exists():
        raise FileNotFoundError("Processed data missing. Run src/data/prepare_musiccaps.py first.")

    label_vocab = json.loads(vocab_path.read_text(encoding="utf-8"))

    train_ds = MusicCapsMelDataset(
        metadata_csv=metadata_csv,
        labels_npy=labels_npy,
        split="train",
        val_ratio=args.val_ratio,
        seed=args.seed,
        target_frames=args.target_frames,
    )
    val_ds = MusicCapsMelDataset(
        metadata_csv=metadata_csv,
        labels_npy=labels_npy,
        split="val",
        val_ratio=args.val_ratio,
        seed=args.seed,
        target_frames=args.target_frames,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MusicLabelCNN(num_labels=len(label_vocab)).to(device)
    pos_weight = compute_pos_weight(train_ds).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    history: list[dict[str, float]] = []
    best_val_f1 = -1.0
    best_val_threshold = args.threshold
    best_path = args.output_dir / "best_model.pt"

    for epoch in range(1, args.epochs + 1):
        train_loss, train_f1, _, _ = run_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            args.threshold,
        )
        val_loss, _, y_true, y_prob = run_epoch(
            model,
            val_loader,
            criterion,
            None,
            device,
            args.threshold,
        )
        val_threshold, val_f1 = threshold_sweep(
            y_true=y_true,
            y_prob=y_prob,
            min_threshold=args.sweep_min_threshold,
            max_threshold=args.sweep_max_threshold,
            step=args.sweep_step,
        )

        row = {
            "epoch": float(epoch),
            "train_loss": train_loss,
            "train_micro_f1": train_f1,
            "val_loss": val_loss,
            "val_micro_f1": val_f1,
            "val_best_threshold": val_threshold,
        }
        history.append(row)
        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} train_f1={train_f1:.4f} | "
            f"val_loss={val_loss:.4f} val_f1={val_f1:.4f} "
            f"(best_threshold={val_threshold:.2f})"
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_val_threshold = val_threshold
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "label_vocab": label_vocab,
                    "args": vars(args),
                    "best_val_micro_f1": best_val_f1,
                    "best_val_threshold": best_val_threshold,
                    "pos_weight": pos_weight.detach().cpu(),
                },
                best_path,
            )

    history_path = args.output_dir / "history.json"
    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")

    summary_path = args.output_dir / "train_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "best_val_micro_f1": best_val_f1,
                "best_val_threshold": best_val_threshold,
                "epochs": args.epochs,
                "train_size": len(train_ds),
                "val_size": len(val_ds),
                "num_labels": len(label_vocab),
                "checkpoint_path": str(best_path),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Saved best checkpoint to {best_path}")
    print(f"Saved training history to {history_path}")
    print(f"Saved summary to {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
