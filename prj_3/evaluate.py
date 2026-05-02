"""
YOLOv8 模型评估脚本
在验证集上计算 mAP、Precision、Recall 等指标。
"""

import argparse
import json
from pathlib import Path

from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="YOLOv8 模型评估")
    parser.add_argument("--model", type=str, required=True,
                        help="训练好的模型权重路径 (如 outputs/train/weights/best.pt)")
    parser.add_argument("--data", type=str, default="VOC.yaml",
                        help="数据集配置 (默认: VOC.yaml)")
    parser.add_argument("--output", type=str, default="outputs/eval",
                        help="评估结果输出目录 (默认: outputs/eval)")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="评估图像尺寸 (默认: 640)")
    parser.add_argument("--batch", type=int, default=16,
                        help="批次大小 (默认: 16)")
    parser.add_argument("--device", type=str, default="auto",
                        help="评估设备 (默认: auto)")
    parser.add_argument("--conf", type=float, default=0.001,
                        help="置信度阈值 (默认: 0.001，评估时使用低阈值)")
    parser.add_argument("--iou", type=float, default=0.6,
                        help="NMS IoU 阈值 (默认: 0.6)")
    return parser.parse_args()


def main():
    args = parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"模型文件不存在: {model_path}")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"加载模型: {model_path}")
    model = YOLO(str(model_path))

    print(f"数据集: {args.data}")

    # 在验证集上评估
    results = model.val(
        data=args.data,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        conf=args.conf,
        iou=args.iou,
        project=str(output_dir),
        name="",
        exist_ok=True,
    )

    # 提取关键指标
    metrics = {
        "mAP@0.5": float(results.box.map50),
        "mAP@0.5:0.95": float(results.box.map),
        "mAP@0.75": float(results.box.map75),
        "precision": float(results.box.mp),
        "recall": float(results.box.mr),
    }

    print(f"\n=== 评估结果 ===")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    # 保存评估结果到 JSON
    result_file = output_dir / "metrics.json"
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print(f"\n评估结果已保存到: {result_file}")

    # 各类别 AP 信息
    print(f"\n=== 各类别 AP@0.5 ===")
    if hasattr(results, 'boxes') and hasattr(results.boxes, 'ap_class_index'):
        ap_per_class = results.boxes.ap50  # shape: [num_classes]
        # 类别名称从 results.names 获取
        names = results.names
        for i, ap in enumerate(ap_per_class):
            class_name = names.get(i, f"class_{i}")
            print(f"  {class_name}: {float(ap):.4f}")


if __name__ == "__main__":
    main()
