from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run zero-shot CLIP classification on CUB-200-2011.")
    parser.add_argument("--config", default="configs/clip_linear_probe_cub.yaml")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--output-dir", default="outputs/zero_shot")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    from src.config import load_config
    from src.data import CUBDataset
    from src.metrics import accuracy
    from src.model import load_open_clip_for_zeroshot

    args = parse_args()
    config = load_config(args.config)
    if args.data_root is not None:
        config["data_root"] = args.data_root

    device = torch.device(args.device)
    model, tokenizer, preprocess = load_open_clip_for_zeroshot(config)
    model = model.to(device).eval()
    dataset = CUBDataset(config["data_root"], "test", preprocess)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=int(config.get("num_workers", 4)))

    prompts = [f"a photo of a {name}, a type of bird." for name in dataset.class_names]
    with torch.no_grad():
        text = tokenizer(prompts).to(device)
        text_features = model.encode_text(text, normalize=True)

    predictions: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    with torch.no_grad():
        for images, labels in tqdm(loader):
            images = images.to(device, non_blocking=True)
            image_features = model.encode_image(images, normalize=True)
            logits = 100.0 * image_features @ text_features.T
            predictions.append(logits.argmax(dim=1).cpu())
            targets.append(labels.cpu())

    pred = torch.cat(predictions)
    target = torch.cat(targets)
    metrics = {"accuracy": accuracy(pred, target), "num_samples": int(target.numel())}
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, ensure_ascii=False)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
