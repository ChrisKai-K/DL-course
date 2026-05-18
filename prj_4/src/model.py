from __future__ import annotations

from dataclasses import dataclass

import open_clip
import torch
from torch import nn

from .peft import freeze_module, inject_adaptformer, inject_lora_by_full_name, mark_classifier_trainable


@dataclass
class BuildResult:
    model: nn.Module
    preprocess_train: object
    preprocess_val: object
    trainable_parameters: int
    total_parameters: int
    injected_modules: int


class CLIPImageClassifier(nn.Module):
    def __init__(self, clip_model: nn.Module, num_classes: int) -> None:
        super().__init__()
        self.clip_model = clip_model
        output_dim = int(getattr(clip_model.visual, "output_dim"))
        self.classifier = nn.Linear(output_dim, num_classes)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.clip_model.encode_image(images, normalize=True)
        return self.classifier(features)


def build_clip_classifier(config: dict) -> BuildResult:
    clip_model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(
        str(config.get("model_name", "ViT-B-32")),
        pretrained=str(config.get("pretrained", "openai")),
    )
    model = CLIPImageClassifier(clip_model, int(config.get("num_classes", 200)))
    freeze_module(model)

    method = str(config.get("method", "linear_probe")).lower()
    injected = 0
    if method == "linear_probe":
        mark_classifier_trainable(model)
    elif method == "lora":
        lora_cfg = dict(config.get("lora", {}))
        target_keywords = list(lora_cfg.get("target_keywords", ["mlp.c_fc", "mlp.c_proj"]))
        injected = inject_lora_by_full_name(
            model.clip_model.visual,
            rank=int(lora_cfg.get("rank", 8)),
            alpha=float(lora_cfg.get("alpha", 16)),
            dropout=float(lora_cfg.get("dropout", 0.1)),
            target_keywords=target_keywords,
        )
        if injected == 0:
            raise RuntimeError(f"LoRA did not match any Linear modules with keywords={target_keywords}")
        mark_classifier_trainable(model)
    elif method == "adaptformer":
        adapter_cfg = dict(config.get("adaptformer", {}))
        injected = inject_adaptformer(
            model.clip_model.visual,
            reduction=int(adapter_cfg.get("reduction", 4)),
            scale=float(adapter_cfg.get("scale", 0.1)),
            dropout=float(adapter_cfg.get("dropout", 0.1)),
            target_keyword=str(adapter_cfg.get("target_keyword", "mlp")),
        )
        if injected == 0:
            raise RuntimeError("AdaptFormer did not match any visual transformer MLP modules.")
        mark_classifier_trainable(model)
    else:
        raise ValueError(f"Unsupported method: {method}")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return BuildResult(model, preprocess_train, preprocess_val, trainable, total, injected)


def load_open_clip_for_zeroshot(config: dict):
    model, _, preprocess_val = open_clip.create_model_and_transforms(
        str(config.get("model_name", "ViT-B-32")),
        pretrained=str(config.get("pretrained", "openai")),
    )
    tokenizer = open_clip.get_tokenizer(str(config.get("model_name", "ViT-B-32")))
    return model, tokenizer, preprocess_val
