from __future__ import annotations

from pathlib import Path

from PIL import Image
from torch.utils.data import DataLoader, Dataset


class CUBDataset(Dataset):
    """CUB-200-2011 split loader using the official metadata files."""

    def __init__(self, data_root: str | Path, split: str, transform) -> None:
        root = Path(data_root)
        self.cub_root = root if root.name == "CUB_200_2011" else root / "CUB_200_2011"
        self.images_dir = self.cub_root / "images"
        self.transform = transform

        if split not in {"train", "test"}:
            raise ValueError("split must be 'train' or 'test'")
        self.split = split

        self.class_names = read_class_names(self.cub_root)
        image_paths = _read_mapping(self.cub_root / "images.txt")
        labels = _read_mapping(self.cub_root / "image_class_labels.txt")
        split_flags = _read_mapping(self.cub_root / "train_test_split.txt")

        wanted_flag = "1" if split == "train" else "0"
        self.samples: list[tuple[Path, int]] = []
        for image_id, relative_path in image_paths.items():
            if split_flags[image_id] != wanted_flag:
                continue
            label = int(labels[image_id]) - 1
            self.samples.append((self.images_dir / relative_path, label))

        if not self.samples:
            raise RuntimeError(f"No CUB samples found for split={split} under {self.cub_root}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, label = self.samples[index]
        image = Image.open(path).convert("RGB")
        return self.transform(image), label


def _read_mapping(path: Path) -> dict[str, str]:
    mapping: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            key, value = line.strip().split(" ", 1)
            mapping[key] = value
    return mapping


def read_class_names(cub_root: str | Path) -> list[str]:
    root = Path(cub_root)
    if root.name != "CUB_200_2011":
        root = root / "CUB_200_2011"
    classes_path = root / "classes.txt"
    if not classes_path.exists():
        return [f"class {i}" for i in range(200)]

    names: list[str] = []
    with classes_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            _, raw_name = line.strip().split(" ", 1)
            name = raw_name.split(".", 1)[-1].replace("_", " ")
            names.append(name)
    return names


def build_dataloaders(config: dict, train_transform, val_transform) -> tuple[DataLoader, DataLoader, list[str]]:
    data_root = config["data_root"]
    batch_size = int(config.get("batch_size", 64))
    num_workers = int(config.get("num_workers", 4))

    train_dataset = CUBDataset(data_root, "train", train_transform)
    val_dataset = CUBDataset(data_root, "test", val_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
    return train_loader, val_loader, train_dataset.class_names
