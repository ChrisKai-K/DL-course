from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict one CUB bird image with a trained CLIP classifier.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--data-root", default=None, help="Needed to read CUB class names.")
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    from src.config import load_config
    from src.data import read_class_names
    from src.model import build_clip_classifier

    args = parse_args()
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    config = load_config(args.config) if args.config else dict(checkpoint["config"])
    if args.data_root is not None:
        config["data_root"] = args.data_root

    build = build_clip_classifier(config)
    model = build.model.to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    class_names = read_class_names(config["data_root"])

    image = Image.open(Path(args.image)).convert("RGB")
    tensor = build.preprocess_val(image).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = model(tensor).softmax(dim=1)[0].cpu()
    values, indices = probs.topk(args.topk)
    for score, index in zip(values.tolist(), indices.tolist(), strict=False):
        print(f"{index:03d} {class_names[index]} {score:.4f}")


if __name__ == "__main__":
    main()
