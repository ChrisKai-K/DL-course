from __future__ import annotations

from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small


def build_mobilenet(num_classes: int, pretrained: bool):
    weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
    model = mobilenet_v3_small(weights=weights)
    model.classifier[3] = model.classifier[3].__class__(
        model.classifier[3].in_features,
        num_classes,
    )
    return model
