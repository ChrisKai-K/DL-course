# 实验三 面向医学图像的分割任务

## 1. 实验目的

本实验基于 FIVES 眼底血管图像数据集，使用 UNet 网络完成像素级血管分割任务。通过本实验，掌握医学图像分割任务的数据预处理、模型训练、指标计算和结果分析方法。

具体目标如下：

1. 了解经典图像分割网络 UNet 的结构特点；
2. 理解眼底图像和血管标注 mask 的预处理流程；
3. 掌握 Accuracy、Recall、Dice 和混淆矩阵等分割指标；
4. 完成模型训练、验证和推理，并分析分割效果。

## 2. 实验环境

| 项目 | 配置 |
|------|------|
| 编程语言 | Python 3 |
| 深度学习框架 | PyTorch 2.0+ |
| 主要依赖 | torch、torchvision、numpy、Pillow、matplotlib、tqdm |
| 推荐硬件 | NVIDIA GPU，或 Apple Silicon MPS / CPU |

## 3. 数据集

本实验使用 FIVES 眼底血管分割数据集。该数据集包含眼底彩色图像及其对应的像素级血管标注，可用于训练医学图像分割模型。

项目中的 `src/data.py` 会递归扫描数据目录，并根据图像和 mask 的文件名进行配对。如果数据集中包含 `train` 和 `test` / `val` 目录，则使用原始划分；否则按指定比例随机划分训练集和验证集。

图像预处理流程如下：

1. 将眼底图像读取为 RGB 三通道；
2. 将血管标注读取为灰度图；
3. 统一缩放到指定尺寸，例如 `512 x 512`；
4. 图像归一化到 `[0, 1]`；
5. mask 二值化，血管像素为 1，背景像素为 0；
6. 训练阶段使用随机水平翻转和垂直翻转增强数据。

## 4. 模型方法

本实验采用 UNet 模型。UNet 是典型的编码器-解码器结构，编码器通过卷积和池化逐步提取高层语义特征，解码器通过上采样逐步恢复空间分辨率。

UNet 的关键设计是跳跃连接。跳跃连接将编码器中较浅层的空间细节特征与解码器中对应尺度的语义特征拼接，使模型能够同时利用局部边缘信息和全局上下文信息。这一特点非常适合医学图像分割任务，尤其适合血管这类细长、边界复杂、像素占比较小的目标。

本项目的 UNet 实现在 `src/model.py` 中，主要模块包括：

| 模块 | 功能 |
|------|------|
| `DoubleConv` | 两层卷积、BatchNorm 和 ReLU |
| `Down` | 最大池化下采样并提取特征 |
| `Up` | 转置卷积上采样并与跳跃连接特征拼接 |
| `UNet` | 完整编码器-解码器网络 |

## 5. 损失函数与评价指标

### 5.1 损失函数

训练阶段使用 BCEWithLogitsLoss 和 Dice Loss 的组合：

```text
Loss = BCEWithLogitsLoss + (1 - Dice)
```

BCE 关注单个像素的二分类正确性，Dice Loss 关注预测区域和真实区域的重叠程度。眼底血管分割中前景血管像素占比较小，加入 Dice Loss 可以缓解类别不平衡问题。

### 5.2 评价指标

本实验计算像素级 TP、TN、FP、FN，并基于这些统计量计算指标：

```text
Accuracy = (TP + TN) / (TP + TN + FP + FN)
Recall = TP / (TP + FN)
Dice = 2TP / (2TP + FP + FN)
IoU = TP / (TP + FP + FN)
```

由于眼底血管属于小目标，背景像素远多于血管像素，因此 Accuracy 可能偏高。分析模型效果时，应重点关注 Dice 和 Recall。

## 6. 实验步骤

### 6.1 安装依赖

```bash
pip install -r requirements.txt
```

### 6.2 训练模型

```bash
python train.py --data data/FIVES --epochs 50 --batch 4 --imgsz 512
```

训练脚本会保存：

| 文件 | 说明 |
|------|------|
| `outputs/checkpoints/best.pt` | 验证集 Dice 最优模型 |
| `outputs/checkpoints/last.pt` | 最后一轮模型 |
| `outputs/history.csv` | 每轮训练和验证指标 |
| `outputs/train_summary.json` | 数据量和最佳 Dice 摘要 |

### 6.3 评估模型

```bash
python evaluate.py --model outputs/checkpoints/best.pt --data data/FIVES
```

评估脚本会输出 Accuracy、Precision、Recall、Dice、IoU，并保存像素级混淆矩阵：

```text
outputs/eval/metrics.json
outputs/eval/confusion_matrix.png
```

### 6.4 可视化分割效果

```bash
python visualize.py --model outputs/checkpoints/best.pt --data data/FIVES --num 6
```

每个样本的可视化结果包含原图、真实 mask、预测概率图和二值分割图，可用于观察模型是否能连续地提取细血管结构。

### 6.5 模型推理

```bash
python infer.py --model outputs/checkpoints/best.pt --source path/to/image_or_folder
```

推理输出包括概率图、二值 mask 和红色叠加图。

## 7. 结果分析

完成训练后，可将 `outputs/eval/metrics.json` 中的结果填写到下表。

| 指标 | 数值 |
|------|------|
| Accuracy | 待填写 |
| Precision | 待填写 |
| Recall | 待填写 |
| Dice | 待填写 |
| IoU | 待填写 |

从分割任务特点来看，血管区域通常较细且占比较低。如果模型预测结果中主干血管较完整，但细小分支缺失，通常会表现为 Recall 和 Dice 不足。若预测结果出现大量背景误分割为血管，则 Precision 会降低。

可能影响分割效果的因素包括：

1. 图像缩放后细血管信息损失；
2. 血管和背景对比度低；
3. 病灶、光照变化和成像模糊带来干扰；
4. 前景血管像素占比小，类别不平衡明显；
5. 训练轮数不足或数据增强不充分。

## 8. 改进方向

后续可从以下方面改进：

1. 增加训练轮数，并观察 `history.csv` 中 Dice 的变化趋势；
2. 使用更高输入分辨率，例如 `768 x 768`；
3. 加入随机亮度、对比度和颜色扰动；
4. 使用带预训练编码器的 UNet，如 ResNet-UNet；
5. 尝试 Attention UNet 或 UNet++；
6. 对 Dice Loss、Focal Loss、Tversky Loss 等损失函数进行对比实验；
7. 对预测结果进行连通域过滤或形态学后处理。

## 9. 实验结论

本实验完成了基于 UNet 的眼底血管图像分割流程，包括数据读取、图像预处理、模型训练、指标评估和结果可视化。UNet 的跳跃连接能够有效保留血管边缘和细节信息，适合作为医学图像分割任务的基础模型。

在实际分析中，应结合 Dice、Recall 和可视化结果综合判断模型性能，而不能只依赖 Accuracy。对于血管这类细小目标，提高细分支召回率和减少背景误分割是后续优化的重点。
