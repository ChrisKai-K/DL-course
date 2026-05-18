from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained CLIP CUB classifier.")
    parser.add_argument("--config", default=None, help="Config path. Defaults to checkpoint config.")
    parser.add_argument("--checkpoint", required=True, help="Path to best_model.pt.")
    parser.add_argument("--data-root", default=None, help="Override CUB root.")
    parser.add_argument("--output-dir", default="outputs/eval", help="Directory for metrics.json.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    from src.config import load_config
    from src.data import build_dataloaders
    from src.engine import evaluate
    from src.model import build_clip_classifier

    args = parse_args()
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    config = load_config(args.config) if args.config else dict(checkpoint["config"])
    if args.data_root is not None:
        config["data_root"] = args.data_root

    build = build_clip_classifier(config)
    _, val_loader, _ = build_dataloaders(config, build.preprocess_train, build.preprocess_val)
    model = build.model.to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    metrics = evaluate(model, val_loader, config, device)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, ensure_ascii=False)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
