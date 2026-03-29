from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .confusion_matrix import build_confusion_matrix, plot_confusion_matrix
from .metrics import compute_accuracy, compute_macro_recall


def _run_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, torch.Tensor, torch.Tensor]:
    is_training = optimizer is not None
    model.train(is_training)

    total_loss = 0.0
    predictions_list: list[torch.Tensor] = []
    targets_list: list[torch.Tensor] = []

    context_manager = torch.enable_grad if is_training else torch.no_grad
    with context_manager():
        for images, targets in tqdm(data_loader, leave=False):
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            logits = model(images)
            loss = criterion(logits, targets)

            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            predictions_list.append(logits.argmax(dim=1).detach().cpu())
            targets_list.append(targets.detach().cpu())

    total_examples = len(data_loader.dataset)
    predictions = torch.cat(predictions_list) if predictions_list else torch.empty(0, dtype=torch.long)
    targets = torch.cat(targets_list) if targets_list else torch.empty(0, dtype=torch.long)
    average_loss = total_loss / max(total_examples, 1)
    return average_loss, predictions, targets


def evaluate_model(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int,
) -> dict[str, Any]:
    loss, predictions, targets = _run_epoch(
        model=model,
        data_loader=data_loader,
        criterion=criterion,
        device=device,
        optimizer=None,
    )
    confusion_matrix = build_confusion_matrix(predictions, targets, num_classes)
    return {
        "loss": loss,
        "accuracy": compute_accuracy(predictions, targets),
        "recall": compute_macro_recall(predictions, targets, num_classes),
        "predictions": predictions,
        "targets": targets,
        "confusion_matrix": confusion_matrix,
    }


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    device: torch.device,
    num_classes: int,
    epochs: int,
    output_dir: Path,
) -> dict[str, Any]:
    history: dict[str, list[float]] = {
        "train_loss": [],
        "val_loss": [],
        "val_accuracy": [],
        "val_recall": [],
    }
    best_state: dict[str, Any] | None = None
    best_accuracy = -1.0

    for epoch in range(1, epochs + 1):
        train_loss, _, _ = _run_epoch(
            model=model,
            data_loader=train_loader,
            criterion=criterion,
            device=device,
            optimizer=optimizer,
        )
        metrics = evaluate_model(
            model=model,
            data_loader=val_loader,
            criterion=criterion,
            device=device,
            num_classes=num_classes,
        )
        if scheduler is not None:
            scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(float(metrics["loss"]))
        history["val_accuracy"].append(float(metrics["accuracy"]))
        history["val_recall"].append(float(metrics["recall"]))

        print(
            f"Epoch {epoch:02d}/{epochs:02d} "
            f"train_loss={train_loss:.4f} "
            f"val_loss={metrics['loss']:.4f} "
            f"val_acc={metrics['accuracy']:.4f} "
            f"val_recall={metrics['recall']:.4f}"
        )

        if float(metrics["accuracy"]) > best_accuracy:
            best_accuracy = float(metrics["accuracy"])
            best_state = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "metrics": {
                    "loss": float(metrics["loss"]),
                    "accuracy": float(metrics["accuracy"]),
                    "recall": float(metrics["recall"]),
                },
            }
            torch.save(best_state, output_dir / "best_model.pt")
            plot_confusion_matrix(
                metrics["confusion_matrix"],
                output_dir / "best_confusion_matrix.png",
                title="Best Validation Confusion Matrix",
            )

    _plot_curves(history, output_dir / "training_curves.png")
    if best_state is None:
        raise RuntimeError("Training finished without producing a checkpoint.")

    return {
        "history": history,
        "best": best_state,
    }


def _plot_curves(history: dict[str, list[float]], save_path: Path) -> None:
    epochs = range(1, len(history["train_loss"]) + 1)
    figure, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(epochs, history["train_loss"], label="Train Loss")
    axes[0].plot(epochs, history["val_loss"], label="Val Loss")
    axes[0].set_title("Loss Curves")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    axes[1].plot(epochs, history["val_accuracy"], label="Accuracy")
    axes[1].plot(epochs, history["val_recall"], label="Recall")
    axes[1].set_title("Validation Metrics")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score")
    axes[1].legend()

    figure.tight_layout()
    figure.savefig(save_path, dpi=200)
    plt.close(figure)
