"""
YOLOv8 目标检测推理脚本
对单张图片或整个文件夹进行目标检测，并保存可视化结果。
"""

import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="YOLOv8 目标检测推理")
    parser.add_argument("--model", type=str, required=True,
                        help="训练好的模型权重路径 (如 outputs/train/weights/best.pt)")
    parser.add_argument("--source", type=str, required=True,
                        help="输入图片路径或文件夹路径")
    parser.add_argument("--output", type=str, default="outputs/detect",
                        help="检测结果输出目录 (默认: outputs/detect)")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="置信度阈值 (默认: 0.25)")
    parser.add_argument("--iou", type=float, default=0.45,
                        help="NMS IoU 阈值 (默认: 0.45)")
    parser.add_argument("--device", type=str, default="auto",
                        help="推理设备 (默认: auto)")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="推理图像尺寸 (默认: 640)")
    parser.add_argument("--save-txt", action="store_true",
                        help="同时保存 txt 格式的检测结果")
    return parser.parse_args()


def main():
    args = parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"模型文件不存在: {model_path}")

    source_path = Path(args.source)
    if not source_path.exists():
        raise FileNotFoundError(f"输入路径不存在: {source_path}")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"加载模型: {model_path}")
    model = YOLO(str(model_path))

    print(f"输入源: {source_path}")
    print(f"输出目录: {output_dir}")
    print(f"置信度阈值: {args.conf}")

    # 执行推理
    results = model.predict(
        source=str(source_path),
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        imgsz=args.imgsz,
        save=True,
        save_txt=args.save_txt,
        project=str(output_dir),
        name="",
        exist_ok=True,
    )

    print(f"\n检测完成！结果保存在: {output_dir}")
    print(f"共处理 {len(results)} 张图片")


if __name__ == "__main__":
    main()
