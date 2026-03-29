from __future__ import annotations

from pathlib import Path

from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class CUBDataset(Dataset):
    def __init__(self, data_root: Path, split: str, transform: transforms.Compose) -> None:
        self.data_root = data_root / "CUB_200_2011"
        self.images_dir = self.data_root / "images"
        self.transform = transform

        image_paths = self._read_mapping(self.data_root / "images.txt")
        labels = self._read_mapping(self.data_root / "image_class_labels.txt")
        train_split = self._read_mapping(self.data_root / "train_test_split.txt")

        split_flag = "1" if split == "train" else "0"
        self.samples: list[tuple[Path, int]] = []

        for image_id, relative_path in image_paths.items():
            if train_split[image_id] != split_flag:
                continue
            label = int(labels[image_id]) - 1
            self.samples.append((self.images_dir / relative_path, label))

    @staticmethod
    def _read_mapping(path: Path) -> dict[str, str]:
        mapping: dict[str, str] = {}
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                key, value = line.strip().split(" ", 1)
                mapping[key] = value
        return mapping

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[object, int]:
        image_path, label = self.samples[index]
        image = Image.open(image_path).convert("RGB")
        return self.transform(image), label


def _build_train_transform(image_size: int, use_augmentation: bool) -> transforms.Compose:
    steps: list[transforms.Transform] = [transforms.Resize((image_size, image_size))]
    if use_augmentation:
        steps.extend(
            [
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            ]
        )
    steps.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
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
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )


def build_cub_dataloaders(
    data_root: Path,
    batch_size: int,
    num_workers: int,
    image_size: int,
    use_augmentation: bool,
) -> tuple[DataLoader, DataLoader]:
    train_dataset = CUBDataset(
        data_root=data_root,
        split="train",
        transform=_build_train_transform(image_size, use_augmentation),
    )
    test_dataset = CUBDataset(
        data_root=data_root,
        split="test",
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
