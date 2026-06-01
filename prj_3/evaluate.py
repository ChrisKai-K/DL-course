from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data import FIVESDataset, find_pairs, split_pairs
from src.metrics import merge_stats, segmentation_stats
from src.model import UNet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained UNet segmentation checkpoint.")
    parser.add_argument("--model", type=str, default="outputs/checkpoints/best.pt", help="Checkpoint path.")
    parser.add_argument("--data", type=str, default="data/FIVES", help="FIVES dataset root.")
    parser.add_argument("--output", type=str, default="outputs/eval", help="Evaluation output directory.")
    parser.add_argument("--batch", type=int, default=4, help="Batch size.")
    parser.add_argument("--imgsz", type=int, default=512, help="Evaluation image size.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Mask probability threshold.")
    parser.add_argument("--device", type=str, default="auto", help="auto, cpu, cuda, or mps.")
    parser.add_argument("--num-workers", type=int, default=2, help="DataLoader workers.")
    parser.add_argument("--seed", type=int, default=42, help="Split seed.")
    return parser.parse_args()


def choose_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def save_confusion_matrix(metrics: dict[str, float], output: Path) -> None:
    matrix = [[metrics["tp"], metrics["fn"]], [metrics["fp"], metrics["tn"]]]
    fig, ax = plt.subplots(figsize=(4, 4), dpi=160)
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks([0, 1], labels=["Pred vessel", "Pred background"], rotation=25, ha="right")
    ax.set_yticks([0, 1], labels=["True vessel", "True background"])
    for y in range(2):
        for x in range(2):
            ax.text(x, y, f"{int(matrix[y][x])}", ha="center", va="center", color="black")
    ax.set_title("Pixel Confusion Matrix")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    checkpoint_path = Path(args.model)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    pairs = find_pairs(args.data)
    _, val_pairs = split_pairs(pairs, seed=args.seed)
    loader = DataLoader(
        FIVESDataset(val_pairs, args.imgsz, augment=False),
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    device = choose_device(args.device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    base_channels = checkpoint.get("args", {}).get("base_channels", 32)
    model = UNet(base_channels=base_channels).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    stats = []
    with torch.no_grad():
        for batch in tqdm(loader):
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            logits = model(images)
            stats.append(segmentation_stats(logits, masks, threshold=args.threshold))

    metrics = merge_stats(stats)
    metrics["threshold"] = args.threshold
    metrics["num_samples"] = len(val_pairs)

    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    save_confusion_matrix(metrics, output_dir / "confusion_matrix.png")

    print("Evaluation metrics:")
    for key in ("accuracy", "precision", "recall", "dice", "iou"):
        print(f"  {key}: {metrics[key]:.4f}")
    print(f"Saved metrics to {output_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
