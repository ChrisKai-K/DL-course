from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate SemFew semantic descriptions for ImageFolder classes.")
    parser.add_argument("--data-root", default="data/miniImageNet")
    parser.add_argument("--split", default="train")
    parser.add_argument("--output", default="outputs/semantic_descriptions.json")
    return parser.parse_args()


def evolve_description(class_name: str) -> str:
    readable = class_name.replace("_", " ").replace("-", " ")
    return (
        f"{readable}: a detailed visual concept. Describe the object category by common shape, "
        f"major parts, colors, surface texture, scene context, scale, pose variation, and visual cues "
        f"that distinguish it from visually similar categories."
    )


def main() -> None:
    args = parse_args()
    split_root = Path(args.data_root) / args.split
    root = split_root if split_root.exists() else Path(args.data_root)
    classes = sorted(path.name for path in root.iterdir() if path.is_dir())
    if not classes:
        raise FileNotFoundError(f"No class directories found under {root}.")
    descriptions = {class_name: evolve_description(class_name) for class_name in classes}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(descriptions, handle, indent=2, ensure_ascii=False)
    print(f"wrote {len(descriptions)} descriptions to {output}")
    print("You can replace these values with WordNet + LLM generated visual descriptions and keep the same JSON keys.")


if __name__ == "__main__":
    main()

