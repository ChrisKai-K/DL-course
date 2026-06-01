# 实验三：面向医学图像的分割任务

本项目按照 `分割实验指导书.docx` 重写，使用 UNet 完成 FIVES 眼底血管图像分割实验。项目包含数据加载、模型训练、验证评估、推理可视化和实验报告模板。

## 实验目标

1. 理解 UNet 编码器-解码器结构和跳跃连接机制。
2. 掌握眼底图像和血管 mask 的预处理。
3. 实现 Accuracy、Recall、Dice、IoU 和混淆矩阵等分割指标。
4. 完成模型训练、验证、推理和结果分析。

## 项目结构

```text
prj_3/
├── train.py
├── evaluate.py
├── infer.py
├── visualize.py
├── src/
│   ├── data.py
│   ├── metrics.py
│   └── model.py
├── reports/
│   └── lab_report.md
├── requirements.txt
└── 分割实验指导书.docx
```

## 数据集放置

下载并解压 FIVES 数据集后，建议放到：

```text
data/FIVES/
```

脚本会递归查找图像和标注 mask，并按文件名自动配对。mask 所在目录名建议包含 `mask`、`label`、`gt`、`Ground truth` 或 `annotation` 等关键词。

如果数据集目录中存在 `train` 和 `test` / `val` / `validation` 文件夹，脚本会优先使用原始划分；否则按 `--val-ratio` 随机划分验证集。

## 环境配置

```bash
pip install -r requirements.txt
```

## 训练

快速测试：

```bash
python train.py --data data/FIVES --epochs 2 --batch 2 --imgsz 256
```

正式实验建议：

```bash
python train.py --data data/FIVES --epochs 50 --batch 4 --imgsz 512
```

主要输出：

```text
outputs/checkpoints/best.pt
outputs/checkpoints/last.pt
outputs/history.csv
outputs/train_summary.json
```

## 评估

```bash
python evaluate.py --model outputs/checkpoints/best.pt --data data/FIVES
```

输出：

```text
outputs/eval/metrics.json
outputs/eval/confusion_matrix.png
```

## 推理

```bash
python infer.py --model outputs/checkpoints/best.pt --source path/to/image_or_folder
```

输出在 `outputs/predict/` 下，包括概率图、二值 mask 和红色叠加图。

## 可视化

```bash
python visualize.py --model outputs/checkpoints/best.pt --data data/FIVES --num 6
```

每张图包含原图、真实 mask、预测概率图和二值预测 mask。

## 常见问题

1. 找不到 mask 文件：检查标注目录名是否包含 `mask`、`label`、`gt`、`Ground truth` 等关键词。
2. 图像和 mask 无法配对：图像和 mask 的文件主名应基本一致，例如 `001.jpg` 和 `001_mask.png`。
3. 显存不足：减小 batch size 或输入尺寸，例如 `python train.py --batch 2 --imgsz 256`。
4. Dice 低但 Accuracy 高：眼底血管像素占比小，背景像素多，分析时应重点关注 Dice 和 Recall。
