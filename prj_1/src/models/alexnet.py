from __future__ import annotations

from torchvision.models import AlexNet_Weights, alexnet


def build_alexnet(num_classes: int, pretrained: bool):
    weights = AlexNet_Weights.DEFAULT if pretrained else None
    model = alexnet(weights=weights)
    model.classifier[6] = model.classifier[6].__class__(
        model.classifier[6].in_features,
        num_classes,
    )
    return model
