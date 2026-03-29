from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import torch


def build_confusion_matrix(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
) -> torch.Tensor:
    matrix = torch.zeros((num_classes, num_classes), dtype=torch.int64)
    for target, prediction in zip(targets.view(-1), predictions.view(-1)):
        matrix[int(target), int(prediction)] += 1
    return matrix


def plot_confusion_matrix(matrix: torch.Tensor, save_path: Path, title: str) -> None:
    figsize = 12 if matrix.shape[0] <= 20 else 18
    figure, axis = plt.subplots(figsize=(figsize, figsize))
    image = axis.imshow(matrix.cpu().numpy(), interpolation="nearest", cmap="Blues")
    axis.set_title(title)
    axis.set_xlabel("Predicted Label")
    axis.set_ylabel("True Label")
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.tight_layout()
    figure.savefig(save_path, dpi=200)
    plt.close(figure)
