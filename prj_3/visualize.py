from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from src.data import FIVESDataset, find_pairs, split_pairs
from src.model import UNet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Save image/mask/prediction comparison panels.")
    parser.add_argument("--model", type=str, default="outputs/checkpoints/best.pt", help="Checkpoint path.")
    parser.add_argument("--data", type=str, default="data/FIVES", help="FIVES dataset root.")
    parser.add_argument("--output", type=str, default="outputs/vis", help="Visualization output directory.")
    parser.add_argument("--imgsz", type=int, default=512, help="Visualization image size.")
    parser.add_argument("--num", type=int, default=6, help="Number of samples.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Mask threshold.")
    parser.add_argument("--device", type=str, default="auto", help="auto, cpu, cuda, or mps.")
    return parser.parse_args()


def choose_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    pairs = find_pairs(args.data)
    _, val_pairs = split_pairs(pairs)
    loader = DataLoader(FIVESDataset(val_pairs[: args.num], args.imgsz, augment=False), batch_size=1, shuffle=False)

    device = choose_device(args.device)
    checkpoint = torch.load(args.model, map_location=device)
    base_channels = checkpoint.get("args", {}).get("base_channels", 32)
    model = UNet(base_channels=base_channels).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    with torch.no_grad():
        for idx, batch in enumerate(loader, start=1):
            image = batch["image"].to(device)
            mask = batch["mask"][0, 0].numpy()
            prob = torch.sigmoid(model(image))[0, 0].cpu().numpy()
            pred = prob >= args.threshold
            image_np = image[0].cpu().permute(1, 2, 0).numpy()

            fig, axes = plt.subplots(1, 4, figsize=(10, 3), dpi=160)
            axes[0].imshow(image_np)
            axes[0].set_title("Image")
            axes[1].imshow(mask, cmap="gray")
            axes[1].set_title("Ground truth")
            axes[2].imshow(prob, cmap="magma", vmin=0, vmax=1)
            axes[2].set_title("Probability")
            axes[3].imshow(pred, cmap="gray")
            axes[3].set_title("Prediction")
            for ax in axes:
                ax.axis("off")
            fig.tight_layout()
            fig.savefig(output_dir / f"sample_{idx:02d}.png")
            plt.close(fig)

    print(f"Saved visualizations to {output_dir}")


if __name__ == "__main__":
    main()
