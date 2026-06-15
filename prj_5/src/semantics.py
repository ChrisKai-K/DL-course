from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

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


class ClipTextEncoder:
    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained: str = "openai",
        checkpoint_path: str | Path | None = None,
    ) -> None:
        self.model_name = model_name
        self.pretrained = pretrained
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
        self.model: Any | None = None
        self.tokenizer: Any | None = None
        self.device: torch.device | None = None

    def _load(self, device: torch.device) -> None:
        if self.model is not None:
            if self.device != device:
                self.model.to(device)
                self.device = device
            return
        try:
            import open_clip
        except ImportError as exc:
            raise ImportError(
                "semantic_encoder='clip' requires open_clip_torch. "
                "Install it with: pip install open_clip_torch"
            ) from exc
        pretrained = str(self.checkpoint_path) if self.checkpoint_path else self.pretrained
        model, _, _ = open_clip.create_model_and_transforms(
            self.model_name,
            pretrained=pretrained,
            device=device,
        )
        model.eval()
        self.model = model
        self.tokenizer = open_clip.get_tokenizer(self.model_name)
        self.device = device

    @torch.no_grad()
    def encode(self, texts: list[str], device: torch.device | str | None = None) -> torch.Tensor:
        target_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self._load(target_device)
        assert self.model is not None
        assert self.tokenizer is not None
        tokens = self.tokenizer(texts).to(target_device)
        embeddings = self.model.encode_text(tokens)
        return F.normalize(embeddings.float(), dim=1)


class SemanticBank:
    def __init__(
        self,
        descriptions: dict[str, str] | None = None,
        dim: int = 512,
        encoder: str = "hash",
        model_name: str = "ViT-B-32",
        pretrained: str = "openai",
        checkpoint_path: str | Path | None = None,
    ) -> None:
        self.descriptions = descriptions or {}
        self.dim = dim
        self.encoder = encoder
        self._cache: dict[str, torch.Tensor] = {}
        self._clip = (
            ClipTextEncoder(model_name=model_name, pretrained=pretrained, checkpoint_path=checkpoint_path)
            if encoder == "clip"
            else None
        )

    @classmethod
    def from_file(
        cls,
        path: str | Path | None,
        dim: int = 512,
        encoder: str = "hash",
        model_name: str = "ViT-B-32",
        pretrained: str = "openai",
        checkpoint_path: str | Path | None = None,
    ) -> "SemanticBank":
        return cls(
            load_semantic_descriptions(path),
            dim=dim,
            encoder=encoder,
            model_name=model_name,
            pretrained=pretrained,
            checkpoint_path=checkpoint_path,
        )

    def describe(self, class_name: str) -> str:
        if class_name in self.descriptions:
            return self.descriptions[class_name]
        normalized = normalize_class_name(class_name)
        return (
            f"{normalized}. A visual category described by its object shape, color, "
            f"texture, parts, background context, and discriminative appearance."
        )

    def encode(self, class_names: list[str], device: torch.device | str | None = None) -> torch.Tensor:
        missing = [name for name in class_names if name not in self._cache]
        if missing:
            descriptions = [self.describe(name) for name in missing]
            if self.encoder == "hash":
                new_embeddings = torch.stack([text_to_embedding(text, self.dim) for text in descriptions])
            elif self.encoder == "clip":
                assert self._clip is not None
                new_embeddings = self._clip.encode(descriptions, device=device).cpu()
                if new_embeddings.size(1) != self.dim:
                    raise ValueError(
                        f"CLIP text embedding dim is {new_embeddings.size(1)}, "
                        f"but config semantic_dim is {self.dim}."
                    )
            else:
                raise ValueError(f"Unsupported semantic_encoder: {self.encoder}")
            for name, embedding in zip(missing, new_embeddings):
                self._cache[name] = embedding.cpu()
        embeddings = torch.stack([self._cache[name] for name in class_names])
        if device is not None:
            embeddings = embeddings.to(device)
        return embeddings
