# 实验二 目标检测（VOC数据集）

## 1.1 实验目的

1. 了解一阶段、二阶段目标检测网络，如 Faster R-CNN、YOLO、SSD、DETR 等；
2. 了解目标检测数据集 VOC 和 COCO，并掌握 Pytorch 环境下不同数据读取方法，为综合实验打基础；
3. 掌握不同目标检测模型的训练、优化方法。

## 1.2 实验要求

1. 了解目标检测网络发展历程；
2. 了解目标检测任务的典型网络结构及改进；
3. 掌握典型算法在目标检测中的应用及其实现；
4. 掌握目标检测算法的模型评估方法。

## 1.3 实验环境

1. 硬件环境：配备 Nvidia 显卡的电脑；
2. 软件环境：Python3 + PyTorch + TorchVision + Ultralytics YOLO + HuggingFace Transformers。

## 1.4 实验内容

本实验旨在通过对比不同目标检测算法，深入理解目标检测技术的核心原理和发展趋势。实验使用 VOC2007 和 COCO2017 数据集，实现并训练四个代表性的目标检测模型：Faster R-CNN（两阶段）、YOLOv8（一阶段）、SSD（一阶段）和 DETR（Transformer-based）。

核心实验内容包括：
1. 搭建统一的数据加载接口，支持 VOC 和 COCO 数据集
2. 实现基于 ResNet50-FPN 的 Faster R-CNN 检测器
3. 集成 Ultralytics YOLOv8 模型
4. 实现 SSD 检测器
5. 集成 DETR (DEtection TRansformer) 模型
6. 统一的训练和评估流程
7. 使用标准 Pascal VOC 指标进行性能评估

## 1.5 实验步骤

### 1. 数据准备

下载并配置 VOC2007 和 COCO2017 数据集。数据集存储在 `./data` 目录下，通过 `datasets/` 模块统一加载。

**VOC 数据集加载实现：**
```python
class VOCDetectionSubset(torchvision.datasets.VOCDetection):
    def __init__(self, root, split, max_images=None, image_size=800):
        super().__init__(root, year='2007', image_set=split)
        self.max_images = max_images
        self.image_size = image_size
        if max_images and len(self) > max_images:
            self._subset_indices = list(range(max_images))
        else:
            self._subset_indices = list(range(len(self)))
```

**COCO 数据集加载实现：**
```python
class COCODetectionSubset(torchvision.datasets.CocoDetection):
    def __init__(self, root, split, max_images=None, image_size=800):
        ann_file = os.path.join(root, f'annotations/instances_{split}2017.json')
        super().__init__(root, ann_file)
        self.max_images = max_images
        self.image_size = image_size
```

### 2. 模型架构实现

#### 2.1 Faster R-CNN

使用 PyTorch 预训练的 Faster R-CNN 模型，基于 ResNet50-FPN 特征提取器。关键代码：

```python
class FasterRCNNDetector(BaseDetector):
    def __init__(self, config):
        super().__init__(config)
        num_classes = config["model"]["num_classes"]
        
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
            weights="DEFAULT"
        )
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
```

训练时，模型输出包括分类损失、边界框回归损失、目标性损失等，总和作为最终训练损失。

#### 2.2 YOLOv8

使用 Ultralytics 库的 YOLOv8 模型。由于 YOLO 有自己的训练流程，需要将数据转换为 YOLO 格式：

```python
def _build_data_yaml(self, train_dataset, val_dataset):
    """Write images to temp dirs and create ultralytics data yaml."""
    tmp_dir = tempfile.mkdtemp(prefix="yolo_data_")
    # 创建训练和验证的图像/标签目录
    # 将 PyTorch 张量转换为 YOLO 格式的标注文件
    # 边界框格式: <class> <center_x> <center_y> <width> <height>
```

#### 2.3 SSD

使用 PyTorch 预训练的 SSD300 模型：

```python
class SSDDetector(BaseDetector):
    def __init__(self, config):
        super().__init__(config)
        num_classes = config["model"]["num_classes"]
        
        self.model = torchvision.models.detection.ssd300_vgg16(
            weights="DEFAULT"
        )
        num_anchors = self.model.head.classification_head.num_anchors
        self.model.head.classification_head.num_classes = num_classes
```

#### 2.4 DETR

使用 HuggingFace 的预训练 DETR 模型：

```python
class DETRDetector(BaseDetector):
    def __init__(self, config):
        super().__init__(config)
        self.model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
        self.model.to(device)
```

### 3. 训练流程

统一的训练入口 `train.py` 根据配置文件加载不同的模型和数据集：

```python
def main():
    config = yaml.safe_load(open(args.config))
    train_dataset = build_dataset(config, split="train")
    val_dataset = build_dataset(config, split="val")
    
    detector = build_model(config)
    
    if model_name == "yolo":
        detector.train_full(train_dataset, val_dataset)
    else:
        for epoch in range(1, epochs + 1):
            metrics = detector.train_one_epoch(train_loader, optimizer, device)
```

训练超参数配置（示例 - Faster R-CNN VOC）：
- Epochs: 5
- Batch Size: 4
- Learning Rate: 0.005
- Momentum: 0.9
- Weight Decay: 0.0005

### 4. 评估流程

评估流程包括：
1. 加载训练好的模型
2. 在验证集上进行推理
3. 将预测结果和真实标注导出为标准格式
4. 使用 Pascal VOC 评估工具计算 mAP

```python
def evaluate(self, dataloader, device):
    self.model.eval()
    results = []
    
    with torch.no_grad():
        for images, targets in dataloader:
            outputs = self.model(images)
            for output, target in zip(outputs, targets):
                results.append({
                    "image_id": target["image_id"],
                    "boxes": output["boxes"],
                    "scores": output["scores"],
                    "labels": output["labels"],
                })
    return results
```

## 1.6 实验结果与分析

### 1.6.1 不同检测方法性能对比

在 VOC2007 和 COCO2017 数据集上，对四种目标检测方法进行了对比实验。实验结果如下：

**VOC2007 数据集 mAP@0.5 结果：**

| 模型 | mAP@0.5 | 类型 |
|------|---------|------|
| Faster R-CNN | 77.49% | 两阶段 |
| SSD | 45.01% | 一阶段 |
| YOLOv8 | 36.12% | 一阶段 |
| DETR | 0.00% | Transformer |

**COCO2017 数据集 mAP@0.5 结果：**

| 模型 | mAP@0.5 | 类型 |
|------|---------|------|
| Faster R-CNN | 52.88% | 两阶段 |
| YOLOv8 | 48.67% | 一阶段 |
| SSD | 19.43% | 一阶段 |
| DETR | 0.00% | Transformer |

### 1.6.2 VOC2007 各类别 AP 详细结果（Faster R-CNN）

| 类别 | AP |
|------|----|
| aeroplane | 94.44% |
| bicycle | 73.91% |
| bird | 68.62% |
| boat | 77.24% |
| bottle | 67.68% |
| bus | 93.33% |
| car | 93.74% |
| cat | 90.53% |
| chair | 73.39% |
| cow | 85.39% |
| diningtable | 48.48% |
| dog | 66.26% |
| horse | 93.69% |
| motorbike | 74.85% |
| person | 86.34% |
| pottedplant | 79.98% |
| sheep | 64.44% |
| sofa | 75.73% |
| train | 60.17% |
| tvmonitor | 81.68% |
| **mAP** | **77.49%** |

### 1.6.3 结果分析

#### 1. 一阶段 vs 两阶段检测器

**Faster R-CNN（两阶段）表现最优：**
- VOC2007: 77.49%
- COCO2017: 52.88%

两阶段检测器通过区域建议网络（RPN）生成候选区域，然后对每个区域进行精确的分类和回归。这种两步设计使得 Faster R-CNN 在精度上具有明显优势。

**YOLOv8（一阶段）在 COCO 上表现良好：**
- VOC2007: 36.12%
- COCO2017: 48.67%

YOLOv8 在 COCO 数据集上的表现接近 Faster R-CNN，显示了现代一阶段检测器的显著进步。YOLOv8 通过 Anchor-free 设计、更高效的损失函数和更深的网络架构，在保持高推理速度的同时提升了精度。

**SSD（一阶段）表现相对较弱：**
- VOC2007: 45.01%
- COCO2017: 19.43%

SSD 在 VOC 数据集上表现尚可，但在更复杂的 COCO 数据集上性能下降明显。这可能与 SSD 相对较旧的网络架构和多尺度检测策略有关。

#### 2. 数据集复杂度对性能的影响

从实验结果可以看出，所有模型在 COCO2017 上的性能都有明显下降：

- Faster R-CNN: 77.49% → 52.88%（下降 24.61%）
- YOLOv8: 36.12% → 48.67%（提升 12.55%）
- SSD: 45.01% → 19.43%（下降 25.58%）

COCO 数据集比 VOC 数据集更复杂：
1. 类别数量更多：VOC 20 类 vs COCO 80 类
2. 小目标比例更高
3. 目标尺度变化更大
4. 遮挡和重叠情况更严重

YOLOv8 在 COCO 上的相对性能提升说明该模型对复杂数据集有更好的适应性。

#### 3. DETR 模型的实验问题

DETR 模型在本实验中 mAP 为 0%，这表明模型未能有效训练或推理。可能的原因：

1. **数据格式不兼容**：DETR 使用特定的数据预处理和标注格式，可能需要额外的适配工作
2. **训练参数不当**：DETR 需要较长的训练时间（通常需要 300 个 epoch）和特定的学习率调度策略
3. **类别映射问题**：DETR 使用 COCO 预训练权重，类别映射需要正确处理
4. **版本兼容性**：HuggingFace Transformers 和原始 DETR 实现可能存在差异

#### 4. 类别性能差异分析

从 Faster R-CNN VOC 结果可以看出，不同类别的 AP 存在显著差异：

**高 AP 类别（>90%）：**
- aeroplane: 94.44%
- bus: 93.33%
- car: 93.74%
- horse: 93.69%

这些类别具有以下特征：
1. 形状特征明显（如飞机、马的轮廓）
2. 尺度较大且相对固定
3. 训练样本分布均匀

**低 AP 类别（<50%）：**
- diningtable: 48.48%
- sheep: 64.44%
- dog: 66.26%

这些类别的挑战：
1. 餐桌形状多样，常被部分遮挡
2. 动物类（羊、狗）姿态变化大
3. 某些类别训练样本较少

### 1.6.4 进一步优化方向

基于实验结果分析，可以从以下方面优化：

#### 1. 模型架构优化

- **YOLOv8**：使用更大的模型（YOLOv8s/l/x）以获得更高的精度
- **SSD**：使用更现代的骨干网络（如 ResNet50）替代 VGG16
- **Faster R-CNN**：增加训练轮数，使用更复杂的 FPN 版本
- **DETR**：正确配置数据预处理，延长训练时间

#### 2. 训练策略优化

- **学习率调度**：使用 Cosine Annealing 或 OneCycle 学习率策略
- **数据增强**：增加更丰富的数据增强（Mosaic、Mixup 等）
- **多尺度训练**：在训练中使用不同的输入尺寸
- **类别平衡**：对少数类别进行过采样或调整损失权重

#### 3. 损失函数优化

- **Focal Loss**：对难样本给予更多关注，缓解正负样本不平衡
- **CIoU/GIoU Loss**：改进边界框回归损失，提高定位精度
- **类别权重**：根据类别样本数量动态调整权重

#### 4. 后处理优化

- **NMS 参数调优**：根据验证集调整置信度阈值和 IoU 阈值
- **多模型集成**：融合多个模型的预测结果
- **测试时增强（TTA）**：在推理时对图像进行翻转、缩放等增强

### 1.6.5 结论

本实验通过对比四种目标检测方法，深入理解了不同架构的特点和适用场景：

1. **两阶段检测器（Faster R-CNN）**在精度上具有优势，适合对准确率要求高的场景
2. **一阶段检测器（YOLOv8）**在速度和精度之间取得了良好平衡，适合实时应用
3. **数据集复杂度**对模型性能有显著影响，需要根据实际应用场景选择合适的模型
4. **训练策略和超参数**对模型性能至关重要，需要针对具体任务进行调优

未来工作将集中在 DETR 模型的正确实现和更全面的性能优化上。
