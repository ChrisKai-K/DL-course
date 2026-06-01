from __future__ import annotations

import random
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
MASK_HINTS = {"mask", "masks", "label", "labels", "gt", "groundtruth", "ground_truth", "annotation", "annotations"}
IMAGE_HINTS = {"image", "images", "original", "originals", "fundus"}


@dataclass(frozen=True)
class SegmentationPair:
    image: Path
    mask: Path


def _normal_key(path: Path) -> str:
    stem = path.stem.lower()
    stem = re.sub(r"(_|-)?(mask|label|manual|gt|groundtruth|ground_truth|annotation)s?$", "", stem)
    stem = re.sub(r"[^a-z0-9]+", "", stem)
    return stem


def _parts(path: Path) -> set[str]:
    return {part.lower().replace(" ", "_") for part in path.parts}


def _is_mask(path: Path) -> bool:
    parts = _parts(path)
    return bool(parts & MASK_HINTS) or any(hint in path.stem.lower() for hint in ("mask", "manual", "gt"))


def find_pairs(root: str | Path) -> list[SegmentationPair]:
    root = Path(root)
    files = [p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTS and p.is_file()]
    masks = [p for p in files if _is_mask(p)]
    images = [p for p in files if not _is_mask(p)]

    if not masks:
        raise FileNotFoundError(
            f"No mask files found under {root}. Put FIVES masks in a folder named like masks, labels, or Ground truth."
        )
    if not images:
        raise FileNotFoundError(f"No image files found under {root}.")

    mask_by_key: dict[str, Path] = {}
    for mask in masks:
        mask_by_key.setdefault(_normal_key(mask), mask)

    pairs: list[SegmentationPair] = []
    for image in images:
        mask = mask_by_key.get(_normal_key(image))
        if mask is not None:
            pairs.append(SegmentationPair(image=image, mask=mask))

    if not pairs:
        sample_images = "\n".join(str(p) for p in images[:5])
        sample_masks = "\n".join(str(p) for p in masks[:5])
        raise RuntimeError(
            "Found images and masks, but could not pair them by file stem.\n"
            f"Sample images:\n{sample_images}\nSample masks:\n{sample_masks}"
        )
    return sorted(pairs, key=lambda x: str(x.image))


def split_pairs(
    pairs: list[SegmentationPair],
    val_ratio: float = 0.2,
    seed: int = 42,
) -> tuple[list[SegmentationPair], list[SegmentationPair]]:
    train_pairs = [p for p in pairs if "train" in _parts(p.image)]
    test_pairs = [p for p in pairs if {"test", "val", "valid", "validation"} & _parts(p.image)]
    if train_pairs and test_pairs:
        return train_pairs, test_pairs

    rng = random.Random(seed)
    shuffled = pairs[:]
    rng.shuffle(shuffled)
    val_size = max(1, int(len(shuffled) * val_ratio))
    return shuffled[val_size:], shuffled[:val_size]


class FIVESDataset(Dataset):
    def __init__(self, pairs: list[SegmentationPair], image_size: int = 512, augment: bool = False):
        self.pairs = pairs
        self.image_size = image_size
        self.augment = augment

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        pair = self.pairs[index]
        image = Image.open(pair.image).convert("RGB")
        mask = Image.open(pair.mask).convert("L")

        image_np = np.asarray(image.resize((self.image_size, self.image_size), Image.BILINEAR), dtype=np.float32) / 255.0
        mask_np = np.asarray(mask.resize((self.image_size, self.image_size), Image.NEAREST), dtype=np.float32)
        mask_np = (mask_np > 127).astype(np.float32)

        if self.augment and random.random() < 0.5:
            image_np = np.ascontiguousarray(np.flip(image_np, axis=1))
            mask_np = np.ascontiguousarray(np.flip(mask_np, axis=1))
        if self.augment and random.random() < 0.5:
            image_np = np.ascontiguousarray(np.flip(image_np, axis=0))
            mask_np = np.ascontiguousarray(np.flip(mask_np, axis=0))

        image_tensor = torch.from_numpy(image_np).permute(2, 0, 1)
        mask_tensor = torch.from_numpy(mask_np).unsqueeze(0)
        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "image_path": str(pair.image),
            "mask_path": str(pair.mask),
        }


def resize_prediction(logits: torch.Tensor, target_hw: tuple[int, int]) -> torch.Tensor:
    return F.interpolate(logits, size=target_hw, mode="bilinear", align_corners=False)
