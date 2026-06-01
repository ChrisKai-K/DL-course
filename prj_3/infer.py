from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from src.model import UNet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run UNet retinal vessel segmentation on images.")
    parser.add_argument("--model", type=str, default="outputs/checkpoints/best.pt", help="Checkpoint path.")
    parser.add_argument("--source", type=str, required=True, help="Image file or directory.")
    parser.add_argument("--output", type=str, default="outputs/predict", help="Prediction output directory.")
    parser.add_argument("--imgsz", type=int, default=512, help="Network input size.")
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


def image_paths(source: Path) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    if source.is_file():
        return [source]
    return sorted(p for p in source.rglob("*") if p.suffix.lower() in exts and p.is_file())


def main() -> None:
    args = parse_args()
    checkpoint_path = Path(args.model)
    source = Path(args.source)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not source.exists():
        raise FileNotFoundError(f"Source not found: {source}")

    device = choose_device(args.device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    base_channels = checkpoint.get("args", {}).get("base_channels", 32)
    model = UNet(base_channels=base_channels).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    paths = image_paths(source)
    if not paths:
        raise FileNotFoundError(f"No images found in {source}")

    with torch.no_grad():
        for path in paths:
            image = Image.open(path).convert("RGB")
            original_size = image.size
            resized = image.resize((args.imgsz, args.imgsz), Image.BILINEAR)
            arr = np.asarray(resized, dtype=np.float32) / 255.0
            tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)
            logits = model(tensor)
            logits = F.interpolate(logits, size=(original_size[1], original_size[0]), mode="bilinear", align_corners=False)
            prob = torch.sigmoid(logits)[0, 0].cpu().numpy()
            mask = (prob >= args.threshold).astype(np.uint8) * 255

            rel_name = path.stem
            Image.fromarray((prob * 255).astype(np.uint8)).save(output_dir / f"{rel_name}_prob.png")
            Image.fromarray(mask).save(output_dir / f"{rel_name}_mask.png")

            overlay = np.asarray(image).copy()
            overlay[mask > 0] = (0.45 * overlay[mask > 0] + 0.55 * np.array([255, 32, 32])).astype(np.uint8)
            Image.fromarray(overlay).save(output_dir / f"{rel_name}_overlay.png")

    print(f"Saved {len(paths)} prediction set(s) to {output_dir}")


if __name__ == "__main__":
    main()
