# 实验一总结

本实验代码对应实验指导书中的图像分类任务，覆盖以下内容：

- 数据集：`CIFAR-10`、`CUB-200-2011`
- 模型：`AlexNet`、`ResNet`、`SENet`、`ViT`
- 指标：`Accuracy`、`Macro Recall`、`Confusion Matrix`
- 输出：训练曲线、最佳模型权重、评估结果

## 建议实验顺序

1. 先运行 `CIFAR-10 + ResNet18`，验证训练流程是否正常。
2. 再比较 `AlexNet`、`SENet`、`ViT` 在 CIFAR-10 上的效果。
3. 最后切换到 `CUB-200-2011`，进行细粒度分类实验。

## 服务器运行建议

```bash
git pull
cd prj_1
python train.py --config configs/cifar10_resnet18.yaml
```

如果服务器已经提前下载好 CUB 数据集，则使用：

```bash
python train.py --config configs/cub_resnet50.yaml --data-root /path/to/cub_root
```
