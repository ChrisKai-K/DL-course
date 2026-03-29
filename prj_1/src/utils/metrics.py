from __future__ import annotations

import torch


def compute_accuracy(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    if targets.numel() == 0:
        return 0.0
    return float((predictions == targets).float().mean().item())


def compute_macro_recall(predictions: torch.Tensor, targets: torch.Tensor, num_classes: int) -> float:
    recalls: list[float] = []
    for class_index in range(num_classes):
        positives = targets == class_index
        positive_count = int(positives.sum().item())
        if positive_count == 0:
            continue
        true_positives = ((predictions == class_index) & positives).sum().item()
        recalls.append(float(true_positives / positive_count))
    if not recalls:
        return 0.0
    return float(sum(recalls) / len(recalls))
