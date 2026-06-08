from __future__ import annotations

import math
from dataclasses import dataclass

import torch


def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    predictions = logits.argmax(dim=1)
    return (predictions == targets).float().mean().item()


@dataclass(frozen=True)
class EpisodeStats:
    mean_accuracy: float
    ci95: float
    episodes: int


def summarize_accuracies(values: list[float]) -> EpisodeStats:
    if not values:
        return EpisodeStats(mean_accuracy=0.0, ci95=0.0, episodes=0)
    mean = sum(values) / len(values)
    if len(values) == 1:
        return EpisodeStats(mean_accuracy=mean, ci95=0.0, episodes=1)
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    ci95 = 1.96 * math.sqrt(variance) / math.sqrt(len(values))
    return EpisodeStats(mean_accuracy=mean, ci95=ci95, episodes=len(values))

