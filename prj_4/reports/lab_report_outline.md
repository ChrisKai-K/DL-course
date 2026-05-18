# 实验四 基于 CLIP 的图像分类

## 4.1 实验目的

本实验通过 CLIP 预训练模型完成 CUB-200-2011 鸟类图像分类，理解 CLIP 的多模态对比学习思想，并掌握 Linear Probe、LoRA、AdaptFormer 等微调方法的训练、验证与推理流程。

## 4.2 实验要求

1. 了解 CLIP 的图像编码器、文本编码器和跨模态特征对齐机制。
2. 掌握基于 CLIP 的鸟类分类任务实现，包括模型选择、训练策略、优化器和超参数设置。
3. 掌握参数高效微调方法在 CLIP 图像编码器上的实现。

## 4.3 实验环境

Python 3、PyTorch 2.x、open_clip_torch、NVIDIA GPU。服务器建议使用 RTX 4090 或 5090，训练时开启 AMP 混合精度。

## 4.4 实验内容

数据集使用 CUB-200-2011，共 11788 张鸟类图片、200 个细粒度类别。实验包括 CLIP 零样本分类、冻结图像编码器的 Linear Probe、以及在 CLIP 视觉编码器中注入 LoRA 或 AdaptFormer 的参数高效微调。

## 4.5 实验步骤

1. 读取 CUB 官方 `images.txt`、`image_class_labels.txt`、`train_test_split.txt` 和 `classes.txt`，构建训练集和测试集。
2. 使用 `open_clip.create_model_and_transforms` 加载 CLIP ViT-B/32 和对应预处理流程。
3. 零样本分类中，将类别名构造成 `a photo of a {class}, a type of bird.`，计算图像特征和文本特征的余弦相似度。
4. Linear Probe 中冻结 CLIP 图像编码器，只训练最后的线性分类层。
5. LoRA 微调中冻结原始 CLIP 参数，在视觉 Transformer 的 MLP 线性层中加入低秩矩阵分支。
6. AdaptFormer 微调中冻结原始 MLP，在 MLP 输出旁增加轻量 Adapter 分支。
7. 每个 epoch 后在测试 split 上计算 accuracy，保存最佳模型、训练曲线和混淆矩阵。

## 4.6 实验结果与分析

将 `outputs/<run_name>/summary.json` 中的 best accuracy 填入本节，并插入 `training_curves.png` 与 `best_confusion_matrix.png`。

调试问题可记录：数据路径不一致时通过 `--data-root` 覆盖；显存不足时降低 batch size；服务器首次运行需下载 CLIP 权重；细粒度鸟类类别相似导致混淆矩阵中近缘类别更容易误分。

结论可写：零样本 CLIP 不需要训练即可完成分类，但 CUB 是细粒度分类任务，类别差异小；LoRA/AdaptFormer 利用训练集标签对图像编码器进行轻量适配，通常能比零样本和单纯 Linear Probe 获得更好的任务适应性，同时训练参数量远低于全量微调。
