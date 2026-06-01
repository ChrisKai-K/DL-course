from __future__ import annotations

import torch
import torch.nn as nn


class DiceBCELoss(nn.Module):
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = self.bce(logits, targets)
        probs = torch.sigmoid(logits)
        probs = probs.view(probs.size(0), -1)
        targets = targets.view(targets.size(0), -1)
        intersection = (probs * targets).sum(dim=1)
        dice = (2.0 * intersection + self.smooth) / (
            probs.sum(dim=1) + targets.sum(dim=1) + self.smooth
        )
        return bce + (1.0 - dice.mean())


@torch.no_grad()
def segmentation_stats(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
) -> dict[str, float]:
    preds = (torch.sigmoid(logits) >= threshold).bool()
    targets = targets.bool()

    tp = (preds & targets).sum().item()
    tn = (~preds & ~targets).sum().item()
    fp = (preds & ~targets).sum().item()
    fn = (~preds & targets).sum().item()
    eps = 1e-7

    return {
        "tp": float(tp),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
        "accuracy": (tp + tn) / (tp + tn + fp + fn + eps),
        "precision": tp / (tp + fp + eps),
        "recall": tp / (tp + fn + eps),
        "dice": (2.0 * tp) / (2.0 * tp + fp + fn + eps),
        "iou": tp / (tp + fp + fn + eps),
    }


def merge_stats(items: list[dict[str, float]]) -> dict[str, float]:
    totals = {
        "tp": sum(x["tp"] for x in items),
        "tn": sum(x["tn"] for x in items),
        "fp": sum(x["fp"] for x in items),
        "fn": sum(x["fn"] for x in items),
    }
    eps = 1e-7
    tp, tn, fp, fn = totals["tp"], totals["tn"], totals["fp"], totals["fn"]
    totals.update(
        {
            "accuracy": (tp + tn) / (tp + tn + fp + fn + eps),
            "precision": tp / (tp + fp + eps),
            "recall": tp / (tp + fn + eps),
            "dice": (2.0 * tp) / (2.0 * tp + fp + fn + eps),
            "iou": tp / (tp + fp + fn + eps),
        }
    )
    return totals
