from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data import FIVESDataset, find_pairs, split_pairs
from src.metrics import DiceBCELoss, merge_stats, segmentation_stats
from src.model import UNet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train UNet for FIVES retinal vessel segmentation.")
    parser.add_argument("--data", type=str, default="data/FIVES", help="FIVES dataset root.")
    parser.add_argument("--output", type=str, default="outputs", help="Output directory.")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs.")
    parser.add_argument("--batch", type=int, default=4, help="Batch size.")
    parser.add_argument("--imgsz", type=int, default=512, help="Square training image size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--base-channels", type=int, default=32, help="UNet base channel width.")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Fallback validation ratio.")
    parser.add_argument("--num-workers", type=int, default=2, help="DataLoader workers.")
    parser.add_argument("--device", type=str, default="auto", help="auto, cpu, cuda, or mps.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def choose_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    stats = []

    with torch.set_grad_enabled(training):
        for batch in tqdm(loader, leave=False):
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            logits = model(images)
            loss = criterion(logits, masks)

            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            stats.append(segmentation_stats(logits.detach(), masks.detach()))

    metrics = merge_stats(stats)
    metrics["loss"] = total_loss / max(1, len(loader.dataset))
    return metrics


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    output_dir = Path(args.output)
    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    pairs = find_pairs(args.data)
    train_pairs, val_pairs = split_pairs(pairs, args.val_ratio, args.seed)
    print(f"Found {len(pairs)} pairs: train={len(train_pairs)}, val={len(val_pairs)}")

    train_loader = DataLoader(
        FIVESDataset(train_pairs, args.imgsz, augment=True),
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        FIVESDataset(val_pairs, args.imgsz, augment=False),
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    device = choose_device(args.device)
    model = UNet(base_channels=args.base_channels).to(device)
    criterion = DiceBCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))

    best_dice = -1.0
    history_path = output_dir / "history.csv"
    with history_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "epoch",
                "train_loss",
                "train_accuracy",
                "train_recall",
                "train_dice",
                "val_loss",
                "val_accuracy",
                "val_recall",
                "val_dice",
                "lr",
            ],
        )
        writer.writeheader()

        for epoch in range(1, args.epochs + 1):
            train_metrics = run_epoch(model, train_loader, criterion, device, optimizer)
            val_metrics = run_epoch(model, val_loader, criterion, device)
            scheduler.step()

            row = {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_accuracy": train_metrics["accuracy"],
                "train_recall": train_metrics["recall"],
                "train_dice": train_metrics["dice"],
                "val_loss": val_metrics["loss"],
                "val_accuracy": val_metrics["accuracy"],
                "val_recall": val_metrics["recall"],
                "val_dice": val_metrics["dice"],
                "lr": optimizer.param_groups[0]["lr"],
            }
            writer.writerow(row)
            f.flush()

            print(
                f"Epoch {epoch:03d}/{args.epochs} "
                f"train_loss={row['train_loss']:.4f} train_dice={row['train_dice']:.4f} "
                f"val_loss={row['val_loss']:.4f} val_dice={row['val_dice']:.4f}"
            )

            state = {"model": model.state_dict(), "args": vars(args), "epoch": epoch, "val_metrics": val_metrics}
            torch.save(state, ckpt_dir / "last.pt")
            if val_metrics["dice"] > best_dice:
                best_dice = val_metrics["dice"]
                torch.save(state, ckpt_dir / "best.pt")

    summary = {
        "best_val_dice": best_dice,
        "pairs": len(pairs),
        "train_pairs": len(train_pairs),
        "val_pairs": len(val_pairs),
    }
    (output_dir / "train_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Training finished. Best checkpoint: {ckpt_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
