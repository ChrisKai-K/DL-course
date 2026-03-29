from __future__ import annotations

from pathlib import Path
from typing import Any

from torch.utils.data import DataLoader

from .cifar10 import build_cifar10_dataloaders
from .cub import build_cub_dataloaders


def _resolve_data_root(dataset_name: str, data_root: str | None) -> Path:
    if data_root is None:
        return Path("data") / dataset_name
    return Path(data_root)


def build_dataloaders(config: dict[str, Any]) -> tuple[DataLoader, DataLoader]:
    dataset_name = str(config["dataset"]).lower()
    data_root = _resolve_data_root(dataset_name, config.get("data_root"))
    batch_size = int(config.get("batch_size", 64))
    num_workers = int(config.get("num_workers", 2))

    if dataset_name == "cifar10":
        return build_cifar10_dataloaders(
            data_root=data_root,
            batch_size=batch_size,
            num_workers=num_workers,
            image_size=int(config.get("image_size", 224)),
            use_augmentation=bool(config.get("use_augmentation", True)),
        )

    if dataset_name in {"cub", "cub_200_2011", "cub-200-2011"}:
        return build_cub_dataloaders(
            data_root=data_root,
            batch_size=batch_size,
            num_workers=num_workers,
            image_size=int(config.get("image_size", 224)),
            use_augmentation=bool(config.get("use_augmentation", True)),
        )

    raise ValueError(f"Unsupported dataset: {config['dataset']}")
