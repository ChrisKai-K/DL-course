from __future__ import annotations

from torchvision.models import VGG16_Weights, vgg16


def build_vgg16(num_classes: int, pretrained: bool):
    weights = VGG16_Weights.DEFAULT if pretrained else None
    model = vgg16(weights=weights)
    model.classifier[6] = model.classifier[6].__class__(
        model.classifier[6].in_features,
        num_classes,
    )
    return model
