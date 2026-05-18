from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a compact markdown summary for the lab report.")
    parser.add_argument("--run-dir", required=True, help="Directory containing summary.json.")
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    with (run_dir / "summary.json").open("r", encoding="utf-8") as handle:
        summary = json.load(handle)
    config = summary["config"]
    best = summary["best"]
    output = Path(args.output) if args.output else run_dir / "report_notes.md"

    text = f"""# 实验四结果记录

## 配置

- 模型：CLIP {config.get("model_name")}，预训练权重 {config.get("pretrained")}
- 微调方法：{config.get("method")}
- 数据集：CUB-200-2011，200 类鸟类细粒度分类
- 训练轮数：{config.get("epochs")}
- Batch size：{config.get("batch_size")}
- 学习率：{config.get("learning_rate")}
- 权重衰减：{config.get("weight_decay")}

## 最优结果

- 最优 epoch：{best["epoch"]}
- 测试集 loss：{best["metrics"]["loss"]:.4f}
- 测试集 accuracy：{best["metrics"]["accuracy"]:.4f}

## 可用于报告的图表

- 训练曲线：`{run_dir / "training_curves.png"}`
- 混淆矩阵：`{run_dir / "best_confusion_matrix.png"}`
- 最优模型：`{run_dir / "best_model.pt"}`

## 分析要点

本实验冻结 CLIP 主干的大部分参数，只训练分类头和少量参数高效微调模块。相比全量微调，这种做法显著降低显存占用和训练时间；相比零样本分类，LoRA/AdaptFormer 能利用 CUB 训练集的细粒度鸟类标签，让视觉编码器的特征更贴近目标数据分布。
"""
    output.write_text(text, encoding="utf-8")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
