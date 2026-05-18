from __future__ import annotations

import math

import torch
from torch import nn


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float) -> None:
        super().__init__()
        self.base = base
        self.rank = rank
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(dropout)
        self.lora_a = nn.Linear(base.in_features, rank, bias=False)
        self.lora_b = nn.Linear(rank, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b.weight)
        for param in self.base.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.lora_b(self.lora_a(self.dropout(x))) * self.scaling


class Adapter(nn.Module):
    def __init__(self, hidden_size: int, reduction: int, scale: float, dropout: float) -> None:
        super().__init__()
        bottleneck = max(hidden_size // reduction, 1)
        self.down = nn.Linear(hidden_size, bottleneck)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.up = nn.Linear(bottleneck, hidden_size)
        self.scale = scale
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(self.dropout(self.act(self.down(x)))) * self.scale


class AdaptedMLP(nn.Module):
    def __init__(self, mlp: nn.Module, hidden_size: int, reduction: int, scale: float, dropout: float) -> None:
        super().__init__()
        self.mlp = mlp
        self.adapter = Adapter(hidden_size, reduction, scale, dropout)
        for param in self.mlp.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x) + self.adapter(x)


def freeze_module(module: nn.Module) -> None:
    for param in module.parameters():
        param.requires_grad = False


def mark_classifier_trainable(model: nn.Module) -> None:
    for name, param in model.named_parameters():
        if name.startswith("classifier."):
            param.requires_grad = True


def inject_lora(module: nn.Module, rank: int, alpha: float, dropout: float, target_keywords: list[str]) -> int:
    replaced = 0
    for child_name, child in list(module.named_children()):
        full_match = any(keyword in child_name for keyword in target_keywords)
        if isinstance(child, nn.Linear) and full_match:
            setattr(module, child_name, LoRALinear(child, rank, alpha, dropout))
            replaced += 1
        else:
            replaced += inject_lora(child, rank, alpha, dropout, target_keywords)
    return replaced


def inject_lora_by_full_name(
    root: nn.Module,
    rank: int,
    alpha: float,
    dropout: float,
    target_keywords: list[str],
) -> int:
    replaced = 0
    for module_name, module in list(root.named_modules()):
        for child_name, child in list(module.named_children()):
            full_name = f"{module_name}.{child_name}" if module_name else child_name
            if isinstance(child, nn.Linear) and any(keyword in full_name for keyword in target_keywords):
                setattr(module, child_name, LoRALinear(child, rank, alpha, dropout))
                replaced += 1
    return replaced


def inject_adaptformer(root: nn.Module, reduction: int, scale: float, dropout: float, target_keyword: str) -> int:
    replaced = 0
    for module_name, module in list(root.named_modules()):
        for child_name, child in list(module.named_children()):
            full_name = f"{module_name}.{child_name}" if module_name else child_name
            if child_name == target_keyword and hasattr(child, "c_fc"):
                hidden_size = int(child.c_fc.in_features)
                setattr(module, child_name, AdaptedMLP(child, hidden_size, reduction, scale, dropout))
                replaced += 1
            elif full_name.endswith(target_keyword) and isinstance(child, nn.Sequential):
                first_linear = next((m for m in child.modules() if isinstance(m, nn.Linear)), None)
                if first_linear is not None:
                    setattr(
                        module,
                        child_name,
                        AdaptedMLP(child, int(first_linear.in_features), reduction, scale, dropout),
                    )
                    replaced += 1
    return replaced
