from __future__ import annotations

import argparse
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a SemFew-style few-shot classifier.")
    parser.add_argument("--config", default="configs/semfew_miniimagenet.yaml")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--semantic-path", default=None)
    parser.add_argument("--semantic-encoder", default=None, choices=["hash", "clip"])
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--val-episodes", type=int, default=None)
    parser.add_argument("--val-interval", type=int, default=None)
    parser.add_argument("--n-way", type=int, default=None)
    parser.add_argument("--k-shot", type=int, default=None)
    parser.add_argument("--q-query", type=int, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main() -> None:
    from src.config import load_config, save_config
    from src.data import EpisodeSampler, FewShotImageFolder, build_transform
    from src.engine import train_semfew
    from src.model import SemFewModel
    from src.semantics import SemanticBank

    args = parse_args()
    config = load_config(args.config)
    if args.data_root is not None:
        config["data_root"] = args.data_root
    if args.semantic_path is not None:
        config["semantic_path"] = args.semantic_path
    if args.semantic_encoder is not None:
        config["semantic_encoder"] = args.semantic_encoder
    if args.episodes is not None:
        config["train_episodes"] = args.episodes
    if args.val_episodes is not None:
        config["val_episodes"] = args.val_episodes
    if args.val_interval is not None:
        config["val_interval"] = args.val_interval
    if args.n_way is not None:
        config["n_way"] = args.n_way
    if args.k_shot is not None:
        config["k_shot"] = args.k_shot
    if args.q_query is not None:
        config["q_query"] = args.q_query

    set_seed(int(config.get("seed", 42)))
    device = torch.device(args.device)

    image_size = int(config.get("image_size", 84))
    train_set = FewShotImageFolder(config["data_root"], "train", build_transform(image_size, train=True))
    val_set = FewShotImageFolder(config["data_root"], "val", build_transform(image_size, train=False))
    sampler_kwargs = {
        "n_way": int(config["n_way"]),
        "k_shot": int(config["k_shot"]),
        "q_query": int(config["q_query"]),
    }
    train_sampler = EpisodeSampler(train_set, seed=int(config.get("seed", 42)), **sampler_kwargs)
    val_sampler = EpisodeSampler(val_set, seed=int(config.get("seed", 42)) + 1, **sampler_kwargs)
    semantic_bank = SemanticBank.from_file(
        config.get("semantic_path"),
        dim=int(config.get("semantic_dim", 512)),
        encoder=str(config.get("semantic_encoder", "hash")),
        model_name=str(config.get("semantic_model", "ViT-B-32")),
        pretrained=str(config.get("semantic_pretrained", "openai")),
        checkpoint_path=config.get("semantic_checkpoint"),
    )
    model = SemFewModel(
        backbone=str(config.get("backbone", "resnet18")),
        pretrained=bool(config.get("pretrained", True)),
        train_backbone=bool(config.get("train_backbone", False)),
        semantic_dim=int(config.get("semantic_dim", 512)),
        hidden_dim=int(config.get("hidden_dim", 512)),
    )

    run_name = str(config.get("run_name") or f"semfew_{datetime.now():%Y%m%d-%H%M%S}")
    output_dir = Path(args.output_dir) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    save_config(config, output_dir / "config.yaml")

    print(f"train_classes={len(train_set.classes)} val_classes={len(val_set.classes)} output_dir={output_dir}")
    summary = train_semfew(model, train_sampler, val_sampler, semantic_bank, config, device, output_dir)
    print(f"best_accuracy={summary['best_accuracy']:.4f}")


if __name__ == "__main__":
    main()
