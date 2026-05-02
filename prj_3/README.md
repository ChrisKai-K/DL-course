# 实验三：目标检测

基于 YOLOv8 的目标检测实验，在 PASCAL VOC 2007 数据集上训练并评估模型。

## 实验目的

1. 理解目标检测任务中卷积神经网络（CNN）的应用
2. 掌握 YOLO 单阶段目标检测方法的原理与实现
3. 学习目标检测模型的评估方法（AP、mAP 等）

## 环境配置

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 验证 GPU（可选但推荐）

```bash
python -c "import torch; print('CUDA 可用:', torch.cuda.is_available())"
```

## 项目结构

```
prj_3/
├── train.py          # 训练脚本
├── detect.py         # 推理检测脚本
├── evaluate.py       # 模型评估脚本
├── requirements.txt  # 依赖包列表
├── README.md         # 本文件
└── outputs/          # 输出目录（训练结果、检测结果、评估指标）
```

## 使用方法

### 1. 训练模型

使用 YOLOv8n（nano 版本，参数量最小，训练最快）在 VOC 2007 数据集上训练：

```bash
python train.py --epochs 5 --batch 16
```

> 首次运行会自动下载 VOC 2007 数据集（约 1.4GB）和 YOLOv8n 预训练权重。
> 如果显存不足，可减小 `--batch` 参数，如 `--batch 8`。

**主要参数说明：**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model` | yolov8n.pt | 模型权重，可选 yolov8s.pt / yolov8m.pt |
| `--data` | VOC.yaml | 数据集配置 |
| `--epochs` | 5 | 训练轮数（快速实验用 5，追求精度可设 20-50） |
| `--batch` | 16 | 批次大小（显存不足时减小） |
| `--imgsz` | 640 | 输入图像尺寸 |
| `--lr0` | 0.01 | 初始学习率 |

### 2. 模型推理

对单张图片或文件夹进行目标检测：

```bash
# 检测单张图片
python detect.py --model outputs/train/weights/best.pt --source /path/to/image.jpg

# 检测整个文件夹
python detect.py --model outputs/train/weights/best.pt --source /path/to/folder/

# 保存 txt 格式结果
python detect.py --model outputs/train/weights/best.pt --source image.jpg --save-txt
```

检测结果（带标注框的图片）默认保存在 `outputs/detect/` 目录。

### 3. 模型评估

在 VOC 2007 验证集上计算 mAP 等指标：

```bash
python evaluate.py --model outputs/train/weights/best.pt
```

评估指标包括：
- **mAP@0.5**：IoU 阈值为 0.5 时的 mean Average Precision
- **mAP@0.5:0.95**：IoU 从 0.5 到 0.95 (步长 0.05) 的平均 mAP
- **Precision**：准确率
- **Recall**：召回率

评估结果保存在 `outputs/eval/metrics.json`。

## 模型原理

### YOLOv8 简介

YOLO（You Only Look Once）是单阶段目标检测算法的代表，核心理念是将目标检测视为回归问题，通过一次前向传播直接输出边界框和类别。

**检测流程：**

1. **网格划分**：将输入图像划分为 S×S 的网格
2. **特征提取**：使用 CSPDarknet 骨干网络提取多尺度特征
3. **特征融合**：通过 PAN-FPN（路径聚合网络-特征金字塔）融合不同层级的特征
4. **检测头**：解耦头（Decoupled Head）分别预测类别和边界框
5. **后处理**：通过 NMS（非极大值抑制）去除重复检测框

### 关键超参数

| 超参数 | 说明 |
|--------|------|
| 学习率 (lr) | 控制参数更新步长，过大震荡不收敛，过小收敛慢 |
| 批次大小 (batch) | 每次迭代使用的样本数，受显存限制 |
| 训练轮数 (epochs) | 遍历数据集的次数，越多拟合越好但可能过拟合 |
| 输入尺寸 (imgsz) | 输入图像大小，越大细节越多但计算量增大 |
| IoU 阈值 | NMS 中判断框重叠程度的阈值 |
| 置信度阈值 (conf) | 过滤低置信度检测框的阈值 |

### 数据增强

YOLOv8 训练时默认使用 Mosaic 增强、随机仿射变换、HSV 颜色空间扰动、水平翻转等数据增强方法，提升模型泛化能力。

### 损失函数

YOLOv8 使用以下损失函数：
- **分类损失**：Binary Cross-Entropy (BCE)
- **边界框回归损失**：CIoU Loss（Complete IoU），同时考虑重叠面积、中心点距离和长宽比
- **DFL 损失**：Distribution Focal Loss，用于提高边界框回归精度

## 实验结果分析

训练完成后，可以从以下角度分析模型性能：

1. **mAP 指标**：观察 mAP@0.5 和 mAP@0.5:0.95，数值越高检测性能越好
2. **各类别 AP**：分析模型在不同类别上的检测能力差异
3. **Precision-Recall 曲线**：反映模型在不同置信度阈值下的表现
4. **检测可视化**：通过 detect.py 查看实际检测效果，观察漏检和误检情况

## 改进方向

1. **增加训练轮数**：将 epochs 从 5 增加到 20-50
2. **使用更大模型**：从 yolov8n 切换到 yolov8s / yolov8m
3. **调整输入尺寸**：增大 imgsz（如 1280）以获得更多细节
4. **超参数调优**：调整学习率、权重衰减等超参数
5. **数据增强优化**：关闭或调整 Mosaic 增强的参数

## YOLO（单阶段）vs Faster R-CNN（两阶段）对比

| 特性 | YOLO (单阶段) | Faster R-CNN (两阶段) |
|------|---------------|------------------------|
| 检测速度 | 快，适合实时场景 | 慢，需先生成候选区域 |
| 检测精度 | 略低，小目标检测偏弱 | 较高，定位更精确 |
| 网络结构 | 端到端单网络 | RPN + 检测网络两阶段 |
| 候选框生成 | 网格直接回归 | RPN 生成候选框后分类 |
| 典型应用 | 实时检测、视频监控 | 高精度检测、医学图像 |

## 常见问题

**1. 显存不足 (Out of Memory)**
```bash
# 减小 batch size
python train.py --batch 8
# 或减小输入尺寸
python train.py --imgsz 320 --batch 8
```

**2. 检测效果不好**
```bash
# 增加训练轮数
python train.py --epochs 20
# 或使用更大的模型
python train.py --model yolov8s.pt --epochs 10
```

**3. 下载数据集太慢**
数据集会自动下载，如需手动下载，可从以下地址获取 VOC 2007/2012 数据集后放置在对应目录。
