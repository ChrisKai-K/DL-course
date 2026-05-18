from __future__ import annotations

import argparse
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CLIP image classifier on CUB-200-2011.")
    parser.add_argument("--config", default="configs/clip_lora_cub.yaml", help="Experiment config path.")
    parser.add_argument("--data-root", default=None, help="Override CUB root containing CUB_200_2011/.")
    parser.add_argument("--output-dir", default="outputs", help="Output root for checkpoints and figures.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=None, help="Override training epochs.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size.")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate.")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main() -> None:
    from src.config import load_config
    from src.data import build_dataloaders
    from src.engine import train
    from src.model import build_clip_classifier

    args = parse_args()
    config = load_config(args.config)
    if args.data_root is not None:
        config["data_root"] = args.data_root
    if args.epochs is not None:
        config["epochs"] = args.epochs
    if args.batch_size is not None:
        config["batch_size"] = args.batch_size
    if args.lr is not None:
        config["learning_rate"] = args.lr

    set_seed(int(config.get("seed", 42)))
    torch.backends.cudnn.benchmark = True
    device = torch.device(args.device)

    build = build_clip_classifier(config)
    train_loader, val_loader, _ = build_dataloaders(config, build.preprocess_train, build.preprocess_val)
    model = build.model.to(device)

    run_name = str(config.get("run_name") or f"clip_{config['method']}_{datetime.now():%Y%m%d-%H%M%S}")
    output_dir = Path(args.output_dir) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"method={config['method']} model={config.get('model_name')} pretrained={config.get('pretrained')}")
    print(f"trainable_parameters={build.trainable_parameters:,} total_parameters={build.total_parameters:,}")
    print(f"injected_modules={build.injected_modules} output_dir={output_dir}")

    summary = train(model, train_loader, val_loader, config, device, output_dir)
    print(f"best_epoch={summary['best']['epoch']} best_acc={summary['best']['metrics']['accuracy']:.4f}")


if __name__ == "__main__":
    main()
