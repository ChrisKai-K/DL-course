from __future__ import annotations

from torchvision.models import ViT_B_16_Weights, vit_b_16


def build_vit(num_classes: int, pretrained: bool):
    weights = ViT_B_16_Weights.DEFAULT if pretrained else None
    model = vit_b_16(weights=weights)
    model.heads.head = model.heads.head.__class__(model.heads.head.in_features, num_classes)
    return model
