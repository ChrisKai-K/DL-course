from __future__ import annotations

from torchvision.models import (
    ResNet18_Weights,
    ResNet34_Weights,
    ResNet50_Weights,
    resnet18,
    resnet34,
    resnet50,
)


def build_resnet(variant: str, num_classes: int, pretrained: bool):
    variant = variant.lower()
    if variant == "resnet18":
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        model = resnet18(weights=weights)
    elif variant == "resnet34":
        weights = ResNet34_Weights.DEFAULT if pretrained else None
        model = resnet34(weights=weights)
    elif variant == "resnet50":
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        model = resnet50(weights=weights)
    else:
        raise ValueError(f"Unsupported ResNet variant: {variant}")

    model.fc = model.fc.__class__(model.fc.in_features, num_classes)
    return model
