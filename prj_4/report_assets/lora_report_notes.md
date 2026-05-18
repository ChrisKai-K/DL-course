# 实验四结果记录

## 配置

- 模型：CLIP ViT-B-32，预训练权重 openai
- 微调方法：lora
- 数据集：CUB-200-2011，200 类鸟类细粒度分类
- 训练轮数：10
- Batch size：64
- 学习率：0.0005
- 权重衰减：0.01

## 最优结果

- 最优 epoch：10
- 测试集 loss：2.8860
- 测试集 accuracy：0.4924

## 可用于报告的图表

- 训练曲线：`outputs/clip_vitb32_lora_cub/training_curves.png`
- 混淆矩阵：`outputs/clip_vitb32_lora_cub/best_confusion_matrix.png`
- 最优模型：`outputs/clip_vitb32_lora_cub/best_model.pt`

## 分析要点

本实验冻结 CLIP 主干的大部分参数，只训练分类头和少量参数高效微调模块。相比全量微调，这种做法显著降低显存占用和训练时间；相比零样本分类，LoRA/AdaptFormer 能利用 CUB 训练集的细粒度鸟类标签，让视觉编码器的特征更贴近目标数据分布。
