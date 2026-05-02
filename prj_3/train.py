"""
YOLOv8 目标检测训练脚本
使用 YOLOv8n (nano) 在 VOC 2007 数据集上训练，训练速度快，适合课程实验。
"""

import argparse
import os
from pathlib import Path

from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="YOLOv8 目标检测训练")
    parser.add_argument("--model", type=str, default="yolov8n.pt",
                        help="YOLO 模型文件 (默认: yolov8n.pt)")
    parser.add_argument("--data", type=str, default="VOC.yaml",
                        help="数据集配置 (默认: VOC.yaml，自动下载 VOC2007)")
    parser.add_argument("--epochs", type=int, default=5,
                        help="训练轮数 (默认: 5，快速实验)")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="输入图像尺寸 (默认: 640)")
    parser.add_argument("--batch", type=int, default=16,
                        help="批次大小 (默认: 16，显存不足时可减小)")
    parser.add_argument("--device", type=str, default="auto",
                        help="训练设备 (默认: auto，自动选择 GPU/CPU)")
    parser.add_argument("--workers", type=int, default=4,
                        help="数据加载线程数 (默认: 4)")
    parser.add_argument("--output", type=str, default="outputs",
                        help="输出目录 (默认: outputs)")
    parser.add_argument("--lr0", type=float, default=0.01,
                        help="初始学习率 (默认: 0.01)")
    return parser.parse_args()


def main():
    args = parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载预训练模型 (yolov8n.pt 首次运行自动下载)
    model = YOLO(args.model)

    print(f"使用设备: {args.device}")
    print(f"数据集: {args.data}")
    print(f"训练轮数: {args.epochs}")
    print(f"批次大小: {args.batch}")
    print(f"输出目录: {output_dir}")

    # 训练
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        lr0=args.lr0,
        project=str(output_dir),
        name="train",
        exist_ok=True,
        pretrained=True,
        # 验证
        val=True,
        # 保存最佳模型
        save=True,
        save_period=1,
    )

    print(f"\n训练完成！最佳模型保存在: {output_dir}/train/weights/best.pt")

    # 打印训练结果摘要
    print(f"\n=== 训练结果摘要 ===")
    if results and hasattr(results, 'results_dict'):
        for k, v in results.results_dict.items():
            print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
