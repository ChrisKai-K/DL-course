from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import torch
import torch.nn.functional as F

TOKEN_RE = re.compile(r"[A-Za-z0-9_\u4e00-\u9fff]+")


def normalize_class_name(name: str) -> str:
    return name.replace("_", " ").replace("-", " ").strip()


def load_semantic_descriptions(path: str | Path | None) -> dict[str, str]:
    if path is None:
        return {}
    semantic_path = Path(path)
    if not semantic_path.exists():
        return {}
    with semantic_path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if isinstance(raw, list):
        return {str(item["class_name"]): str(item["description"]) for item in raw}
    return {str(key): str(value) for key, value in raw.items()}


def text_to_embedding(text: str, dim: int = 512) -> torch.Tensor:
    vector = torch.zeros(dim, dtype=torch.float32)
    tokens = TOKEN_RE.findall(text.lower())
    for token in tokens:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        bucket = int.from_bytes(digest[:4], "little") % dim
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vector[bucket] += sign
    if not tokens:
        vector[0] = 1.0
    return F.normalize(vector, dim=0)


class SemanticBank:
    def __init__(self, descriptions: dict[str, str] | None = None, dim: int = 512) -> None:
        self.descriptions = descriptions or {}
        self.dim = dim

    @classmethod
    def from_file(cls, path: str | Path | None, dim: int = 512) -> "SemanticBank":
        return cls(load_semantic_descriptions(path), dim=dim)

    def describe(self, class_name: str) -> str:
        if class_name in self.descriptions:
            return self.descriptions[class_name]
        normalized = normalize_class_name(class_name)
        return (
            f"{normalized}. A visual category described by its object shape, color, "
            f"texture, parts, background context, and discriminative appearance."
        )

    def encode(self, class_names: list[str], device: torch.device | str | None = None) -> torch.Tensor:
        embeddings = torch.stack([text_to_embedding(self.describe(name), self.dim) for name in class_names])
        if device is not None:
            embeddings = embeddings.to(device)
        return embeddings

