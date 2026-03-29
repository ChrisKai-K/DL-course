from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate classification models for project 1.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/cifar10_resnet18.yaml",
        help="Path to the experiment config file.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to a saved checkpoint.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Evaluation device.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/eval",
        help="Directory for evaluation artifacts.",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=None,
        help="Optional dataset root override.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from src.config import load_config
    from src.datasets import build_dataloaders
    from src.models import build_model
    from src.utils import evaluate_model, plot_confusion_matrix

    config = load_config(args.config)
    if args.data_root is not None:
        config["data_root"] = args.data_root
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    _, test_loader = build_dataloaders(config)
    model = build_model(config).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    metrics = evaluate_model(
        model=model,
        data_loader=test_loader,
        criterion=nn.CrossEntropyLoss(),
        device=device,
        num_classes=int(config["num_classes"]),
    )

    plot_confusion_matrix(
        metrics["confusion_matrix"],
        output_dir / "confusion_matrix.png",
        title="Evaluation Confusion Matrix",
    )

    serializable_metrics = {
        "loss": float(metrics["loss"]),
        "accuracy": float(metrics["accuracy"]),
        "recall": float(metrics["recall"]),
    }
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(serializable_metrics, handle, indent=2, ensure_ascii=False)

    print(json.dumps(serializable_metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
