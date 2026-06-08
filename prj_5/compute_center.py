from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute visual and semantic class centers.")
    parser.add_argument("--config", default="configs/semfew_miniimagenet.yaml")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--semantic-path", default=None)
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--output", default="outputs/centers.pt")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    from src.config import load_config
    from src.data import FewShotImageFolder, build_transform
    from src.model import ResNetFeatureExtractor
    from src.semantics import SemanticBank

    args = parse_args()
    config = load_config(args.config)
    if args.data_root is not None:
        config["data_root"] = args.data_root
    if args.semantic_path is not None:
        config["semantic_path"] = args.semantic_path
    device = torch.device(args.device)
    dataset = FewShotImageFolder(
        config["data_root"], args.split, build_transform(int(config.get("image_size", 84)), train=False)
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    encoder = ResNetFeatureExtractor(
        str(config.get("backbone", "resnet18")),
        pretrained=bool(config.get("pretrained", True)),
        train_backbone=False,
    ).to(device)
    encoder.eval()

    sums = {class_name: torch.zeros(encoder.feature_dim) for class_name in dataset.classes}
    counts = {class_name: 0 for class_name in dataset.classes}
    for images, _, class_names in tqdm(loader, desc="visual centers", dynamic_ncols=True):
        features = encoder(images.to(device)).cpu()
        for feature, class_name in zip(features, class_names):
            sums[class_name] += feature
            counts[class_name] += 1
    visual_centers = {name: F.normalize(sums[name] / counts[name], dim=0) for name in dataset.classes}

    semantic_bank = SemanticBank.from_file(config.get("semantic_path"), dim=int(config.get("semantic_dim", 512)))
    semantic_centers = semantic_bank.encode(dataset.classes).cpu()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "classes": dataset.classes,
            "visual_centers": torch.stack([visual_centers[name] for name in dataset.classes]),
            "semantic_centers": semantic_centers,
            "counts": counts,
        },
        output,
    )
    print(json.dumps({"classes": len(dataset.classes), "output": str(output)}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

