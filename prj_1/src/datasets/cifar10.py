from __future__ import annotations

from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def _build_train_transform(image_size: int, use_augmentation: bool) -> transforms.Compose:
    steps: list[transforms.Transform] = [transforms.Resize((image_size, image_size))]
    if use_augmentation:
        steps.extend(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(image_size, padding=8),
            ]
        )
    steps.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.4914, 0.4822, 0.4465),
                std=(0.2023, 0.1994, 0.2010),
            ),
        ]
    )
    return transforms.Compose(steps)


def _build_eval_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.4914, 0.4822, 0.4465),
                std=(0.2023, 0.1994, 0.2010),
            ),
        ]
    )


def build_cifar10_dataloaders(
    data_root: Path,
    batch_size: int,
    num_workers: int,
    image_size: int,
    use_augmentation: bool,
) -> tuple[DataLoader, DataLoader]:
    train_dataset = datasets.CIFAR10(
        root=str(data_root),
        train=True,
        download=True,
        transform=_build_train_transform(image_size, use_augmentation),
    )
    test_dataset = datasets.CIFAR10(
        root=str(data_root),
        train=False,
        download=True,
        transform=_build_eval_transform(image_size),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, test_loader
