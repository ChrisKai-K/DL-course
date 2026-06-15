from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a SemFew checkpoint on few-shot episodes.")
    parser.add_argument("--checkpoint", required=True, help="Path to best_model.pt.")
    parser.add_argument("--config", default=None, help="Optional config override.")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--semantic-path", default=None)
    parser.add_argument("--semantic-encoder", default=None, choices=["hash", "clip"])
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--output-dir", default="outputs/test")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    from src.config import load_config
    from src.data import EpisodeSampler, FewShotImageFolder, build_transform
    from src.engine import evaluate_semfew
    from src.model import SemFewModel
    from src.semantics import SemanticBank

    args = parse_args()
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    config = load_config(args.config) if args.config else dict(checkpoint["config"])
    if args.data_root is not None:
        config["data_root"] = args.data_root
    if args.semantic_path is not None:
        config["semantic_path"] = args.semantic_path
    if args.semantic_encoder is not None:
        config["semantic_encoder"] = args.semantic_encoder

    dataset = FewShotImageFolder(
        config["data_root"], args.split, build_transform(int(config.get("image_size", 84)), train=False)
    )
    sampler = EpisodeSampler(
        dataset,
        n_way=int(config["n_way"]),
        k_shot=int(config["k_shot"]),
        q_query=int(config["q_query"]),
        seed=int(config.get("seed", 42)) + 2,
    )
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
        pretrained=False,
        train_backbone=bool(config.get("train_backbone", False)),
        semantic_dim=int(config.get("semantic_dim", 512)),
        hidden_dim=int(config.get("hidden_dim", 512)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    episodes = int(args.episodes or config.get("test_episodes", 600))
    stats = evaluate_semfew(model, sampler, semantic_bank, episodes, device)
    result = {
        "split": args.split,
        "episodes": stats.episodes,
        "accuracy": stats.mean_accuracy,
        "ci95": stats.ci95,
        "n_way": int(config["n_way"]),
        "k_shot": int(config["k_shot"]),
        "q_query": int(config["q_query"]),
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
