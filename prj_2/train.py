import argparse
import os
import yaml
import torch
from torch.utils.data import DataLoader

from datasets import build_dataset
from models import build_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to yaml config file")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = torch.device(config["training"].get("device", "cuda")
                          if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_dataset = build_dataset(config, split="train")
    val_dataset = build_dataset(config, split="val")
    print(f"Train: {len(train_dataset)} images, Val: {len(val_dataset)} images")

    model_name = config["model"]["name"]
    detector = build_model(config)

    # YOLO has its own training loop
    if model_name == "yolo":
        detector.train_full(train_dataset, val_dataset)
        ckpt_dir = config["output"]["checkpoint_dir"]
        detector.save(os.path.join(ckpt_dir, "checkpoint.pt"))
        print(f"YOLO training done. Checkpoint saved to {ckpt_dir}")
        return

    batch_size = config["training"].get("batch_size", 4)
    num_workers = config["training"].get("num_workers", 4)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=train_dataset.collate_fn,
    )

    optimizer = detector.get_optimizer()
    epochs = config["training"].get("epochs", 5)
    ckpt_dir = config["output"]["checkpoint_dir"]
    os.makedirs(ckpt_dir, exist_ok=True)

    for epoch in range(1, epochs + 1):
        metrics = detector.train_one_epoch(train_loader, optimizer, device)
        print(f"Epoch {epoch}/{epochs} — loss: {metrics['loss']:.4f}")

    ckpt_path = os.path.join(ckpt_dir, "checkpoint.pt")
    detector.save(ckpt_path)
    print(f"Saved checkpoint to {ckpt_path}")


if __name__ == "__main__":
    main()
