from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import torch
from torch import nn


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train classification models for project 1.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/cifar10_resnet18.yaml",
        help="Path to the experiment config file.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Root directory for checkpoints and figures.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Training device.",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=None,
        help="Optional dataset root override.",
    )
    return parser.parse_args()


def build_optimizer(model: nn.Module, config: dict[str, object]) -> torch.optim.Optimizer:
    optimizer_name = str(config.get("optimizer", "adam")).lower()
    learning_rate = float(config.get("learning_rate", 1e-3))
    weight_decay = float(config.get("weight_decay", 0.0))

    if optimizer_name == "sgd":
        momentum = float(config.get("momentum", 0.9))
        return torch.optim.SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay,
        )
    if optimizer_name == "adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
    raise ValueError(f"Unsupported optimizer: {optimizer_name}")


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    config: dict[str, object],
) -> torch.optim.lr_scheduler.LRScheduler | None:
    scheduler_name = str(config.get("scheduler", "none")).lower()
    if scheduler_name == "none":
        return None
    if scheduler_name == "step":
        step_size = int(config.get("step_size", 5))
        gamma = float(config.get("gamma", 0.1))
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    raise ValueError(f"Unsupported scheduler: {scheduler_name}")


def main() -> None:
    args = parse_args()
    from src.config import load_config
    from src.datasets import build_dataloaders
    from src.models import build_model
    from src.utils import train_model

    config = load_config(args.config)
    if args.data_root is not None:
        config["data_root"] = args.data_root
    device = torch.device(args.device)

    run_name = config.get("run_name")
    if not run_name:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        run_name = f"{config['dataset']}_{config['model']}_{timestamp}"

    output_dir = Path(args.output_dir) / str(run_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader = build_dataloaders(config)
    model = build_model(config).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config)

    results = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        num_classes=int(config["num_classes"]),
        epochs=int(config.get("epochs", 10)),
        output_dir=output_dir,
    )

    summary = {
        "config": config,
        "best_epoch": results["best"]["epoch"],
        "best_metrics": results["best"]["metrics"],
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    print(f"Training complete. Outputs saved to {output_dir}")


if __name__ == "__main__":
    main()
