from __future__ import annotations

from typing import Any

from .alexnet import build_alexnet
from .resnet import build_resnet
from .senet import build_senet
from .vit import build_vit


def build_model(config: dict[str, Any]):
    model_name = str(config["model"]).lower()
    num_classes = int(config["num_classes"])
    pretrained = bool(config.get("pretrained", False))

    if model_name == "alexnet":
        return build_alexnet(num_classes=num_classes, pretrained=pretrained)
    if model_name.startswith("resnet"):
        return build_resnet(
            variant=model_name,
            num_classes=num_classes,
            pretrained=pretrained,
        )
    if model_name in {"senet", "seresnet18", "seresnet34"}:
        return build_senet(
            variant=model_name,
            num_classes=num_classes,
        )
    if model_name in {"vit", "vit_b_16"}:
        return build_vit(num_classes=num_classes, pretrained=pretrained)

    raise ValueError(f"Unsupported model: {config['model']}")
