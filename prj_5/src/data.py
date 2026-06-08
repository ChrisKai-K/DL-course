from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def resolve_split_root(data_root: str | Path, split: str) -> Path:
    root = Path(data_root)
    candidate = root / split
    return candidate if candidate.exists() else root


def build_transform(image_size: int = 84, train: bool = True) -> transforms.Compose:
    if train:
        return transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )


class FewShotImageFolder(Dataset):
    def __init__(self, data_root: str | Path, split: str = "train", transform=None) -> None:
        self.root = resolve_split_root(data_root, split)
        self.transform = transform
        self.class_to_images: dict[str, list[Path]] = {}
        for class_dir in sorted(path for path in self.root.iterdir() if path.is_dir()):
            images = sorted(path for path in class_dir.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS)
            if images:
                self.class_to_images[class_dir.name] = images
        if not self.class_to_images:
            raise FileNotFoundError(
                f"No ImageFolder classes found under {self.root}. Expected class_name/*.jpg layout."
            )
        self.classes = sorted(self.class_to_images)
        self.samples = [(image, class_name) for class_name in self.classes for image in self.class_to_images[class_name]]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, str]:
        image_path, class_name = self.samples[index]
        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, self.classes.index(class_name), class_name

    def image_count(self, class_name: str) -> int:
        return len(self.class_to_images[class_name])


@dataclass
class Episode:
    support_images: torch.Tensor
    support_labels: torch.Tensor
    query_images: torch.Tensor
    query_labels: torch.Tensor
    class_names: list[str]


class EpisodeSampler:
    def __init__(
        self,
        dataset: FewShotImageFolder,
        n_way: int,
        k_shot: int,
        q_query: int,
        seed: int | None = None,
    ) -> None:
        self.dataset = dataset
        self.n_way = n_way
        self.k_shot = k_shot
        self.q_query = q_query
        self.random = random.Random(seed)
        eligible = [name for name in dataset.classes if dataset.image_count(name) >= k_shot + q_query]
        if len(eligible) < n_way:
            raise ValueError(
                f"Need at least {n_way} classes with {k_shot + q_query} images each; found {len(eligible)}."
            )
        self.eligible_classes = eligible

    def sample(self) -> Episode:
        class_names = self.random.sample(self.eligible_classes, self.n_way)
        support_images: list[torch.Tensor] = []
        query_images: list[torch.Tensor] = []
        support_labels: list[int] = []
        query_labels: list[int] = []

        for label, class_name in enumerate(class_names):
            paths = self.random.sample(self.dataset.class_to_images[class_name], self.k_shot + self.q_query)
            support_paths = paths[: self.k_shot]
            query_paths = paths[self.k_shot :]
            support_images.extend(self._load(path) for path in support_paths)
            query_images.extend(self._load(path) for path in query_paths)
            support_labels.extend([label] * self.k_shot)
            query_labels.extend([label] * self.q_query)

        return Episode(
            support_images=torch.stack(support_images),
            support_labels=torch.tensor(support_labels, dtype=torch.long),
            query_images=torch.stack(query_images),
            query_labels=torch.tensor(query_labels, dtype=torch.long),
            class_names=class_names,
        )

    def _load(self, path: Path) -> torch.Tensor:
        image = Image.open(path).convert("RGB")
        if self.dataset.transform is None:
            raise ValueError("Dataset transform must return tensors for episodic sampling.")
        return self.dataset.transform(image)

