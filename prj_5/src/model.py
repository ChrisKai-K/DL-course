from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F
from torchvision import models


class ResNetFeatureExtractor(nn.Module):
    def __init__(self, name: str = "resnet18", pretrained: bool = True, train_backbone: bool = False) -> None:
        super().__init__()
        if name != "resnet18":
            raise ValueError("This experiment implementation currently supports resnet18.")
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        backbone = models.resnet18(weights=weights)
        self.feature_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        if not train_backbone:
            for parameter in self.backbone.parameters():
                parameter.requires_grad = False

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.backbone(images), dim=1)


class SemAlign(nn.Module):
    def __init__(self, visual_dim: int, semantic_dim: int, hidden_dim: int = 512, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(visual_dim + semantic_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, visual_dim),
        )

    def forward(self, visual_centers: torch.Tensor, semantic_centers: torch.Tensor) -> torch.Tensor:
        fused = torch.cat([visual_centers, semantic_centers], dim=1)
        residual = self.net(fused)
        return F.normalize(visual_centers + residual, dim=1)


@dataclass
class SemFewOutput:
    logits: torch.Tensor
    prototypes: torch.Tensor
    support_centers: torch.Tensor


class SemFewModel(nn.Module):
    def __init__(
        self,
        backbone: str = "resnet18",
        pretrained: bool = True,
        train_backbone: bool = False,
        semantic_dim: int = 512,
        hidden_dim: int = 512,
        temperature: float = 10.0,
    ) -> None:
        super().__init__()
        self.encoder = ResNetFeatureExtractor(backbone, pretrained=pretrained, train_backbone=train_backbone)
        self.align = SemAlign(self.encoder.feature_dim, semantic_dim, hidden_dim=hidden_dim)
        self.log_temperature = nn.Parameter(torch.log(torch.tensor(float(temperature))))

    @property
    def feature_dim(self) -> int:
        return self.encoder.feature_dim

    def forward(
        self,
        support_images: torch.Tensor,
        support_labels: torch.Tensor,
        query_images: torch.Tensor,
        semantic_centers: torch.Tensor,
    ) -> SemFewOutput:
        support_features = self.encoder(support_images)
        query_features = self.encoder(query_images)
        support_centers = compute_centers(support_features, support_labels, semantic_centers.size(0))
        prototypes = self.align(support_centers, semantic_centers)
        logits = self.log_temperature.exp().clamp(max=100.0) * query_features @ prototypes.t()
        return SemFewOutput(logits=logits, prototypes=prototypes, support_centers=support_centers)


def compute_centers(features: torch.Tensor, labels: torch.Tensor, n_classes: int) -> torch.Tensor:
    centers = []
    for label in range(n_classes):
        mask = labels == label
        if not torch.any(mask):
            raise ValueError(f"Class {label} has no support examples.")
        centers.append(features[mask].mean(dim=0))
    return F.normalize(torch.stack(centers), dim=1)

