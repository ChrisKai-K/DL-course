from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .metrics import accuracy, confusion_matrix


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.cuda.amp.GradScaler | None = None,
    amp: bool = True,
) -> dict[str, Any]:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    predictions: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []

    context = torch.enable_grad if is_train else torch.no_grad
    with context():
        for images, labels in tqdm(loader, leave=False):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with torch.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
                logits = model(images)
                loss = criterion(logits, labels)

            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None and scaler.is_enabled():
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

            total_loss += float(loss.item()) * images.size(0)
            predictions.append(logits.argmax(dim=1).detach().cpu())
            targets.append(labels.detach().cpu())

    pred_tensor = torch.cat(predictions) if predictions else torch.empty(0, dtype=torch.long)
    target_tensor = torch.cat(targets) if targets else torch.empty(0, dtype=torch.long)
    return {
        "loss": total_loss / max(len(loader.dataset), 1),
        "accuracy": accuracy(pred_tensor, target_tensor),
        "predictions": pred_tensor,
        "targets": target_tensor,
    }


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: dict,
    device: torch.device,
    output_dir: Path,
) -> dict[str, Any]:
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(config.get("learning_rate", 5e-4)),
        weight_decay=float(config.get("weight_decay", 0.01)),
    )
    epochs = int(config.get("epochs", 10))
    warmup_epochs = int(config.get("warmup_epochs", 0))

    def lr_factor(epoch: int) -> float:
        if warmup_epochs > 0 and epoch < warmup_epochs:
            return float(epoch + 1) / float(warmup_epochs)
        return 1.0

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_factor)
    scaler = torch.cuda.amp.GradScaler(enabled=bool(config.get("amp", True)) and device.type == "cuda")

    history = {"train_loss": [], "train_accuracy": [], "val_loss": [], "val_accuracy": []}
    best: dict[str, Any] | None = None
    best_acc = -1.0

    for epoch in range(1, epochs + 1):
        train_metrics = run_epoch(
            model, train_loader, criterion, device, optimizer=optimizer, scaler=scaler, amp=bool(config.get("amp", True))
        )
        val_metrics = run_epoch(model, val_loader, criterion, device, amp=bool(config.get("amp", True)))
        scheduler.step()

        history["train_loss"].append(float(train_metrics["loss"]))
        history["train_accuracy"].append(float(train_metrics["accuracy"]))
        history["val_loss"].append(float(val_metrics["loss"]))
        history["val_accuracy"].append(float(val_metrics["accuracy"]))

        print(
            f"Epoch {epoch:02d}/{epochs:02d} "
            f"train_loss={train_metrics['loss']:.4f} "
            f"train_acc={train_metrics['accuracy']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_acc={val_metrics['accuracy']:.4f}"
        )

        if float(val_metrics["accuracy"]) > best_acc:
            best_acc = float(val_metrics["accuracy"])
            best = {
                "epoch": epoch,
                "metrics": {"loss": float(val_metrics["loss"]), "accuracy": float(val_metrics["accuracy"])},
            }
            torch.save(
                {
                    "epoch": epoch,
                    "config": config,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "metrics": best["metrics"],
                },
                output_dir / "best_model.pt",
            )
            plot_confusion_matrix(
                confusion_matrix(
                    val_metrics["predictions"], val_metrics["targets"], int(config.get("num_classes", 200))
                ),
                output_dir / "best_confusion_matrix.png",
            )

    if best is None:
        raise RuntimeError("Training produced no checkpoint.")

    plot_curves(history, output_dir / "training_curves.png")
    summary = {"config": config, "best": best, "history": history}
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    return summary


def evaluate(model: nn.Module, loader: DataLoader, config: dict, device: torch.device) -> dict[str, float]:
    metrics = run_epoch(model, loader, nn.CrossEntropyLoss(), device, amp=bool(config.get("amp", True)))
    return {"loss": float(metrics["loss"]), "accuracy": float(metrics["accuracy"])}


def plot_curves(history: dict[str, list[float]], path: Path) -> None:
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(epochs, history["train_loss"], label="Train")
    axes[0].plot(epochs, history["val_loss"], label="Validation")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[1].plot(epochs, history["train_accuracy"], label="Train")
    axes[1].plot(epochs, history["val_accuracy"], label="Validation")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_confusion_matrix(matrix: torch.Tensor, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 8))
    image = ax.imshow(matrix.numpy(), cmap="Blues")
    ax.set_title("CUB-200 Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
