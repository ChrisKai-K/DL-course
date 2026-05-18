# 实验四：基于 CLIP 的图像分类

本项目基于 CLIP 完成 CUB-200-2011 鸟类细粒度分类，覆盖实验指导书中的零样本分类、Linear Probe、LoRA 微调和 AdaptFormer 微调。

## 环境配置

```bash
cd prj_4
pip install -r requirements.txt
```

建议在 4090/5090 服务器上使用 Python 3.10+、PyTorch 2.x、CUDA 版 torch。首次运行会自动下载 CLIP 预训练权重。

## 数据准备

将 CUB-200-2011 解压为以下结构：

```text
data/cub_200_2011/CUB_200_2011/
  images/
  images.txt
  image_class_labels.txt
  train_test_split.txt
  classes.txt
```

如果数据在服务器其他位置，所有脚本都可以通过 `--data-root /path/to/cub_200_2011` 指定。

## 快速运行

上传到服务器前可以先打包：

```bash
bash scripts/pack_for_server.sh
```

服务器解压后进入 `prj_4` 目录安装依赖并运行训练。

零样本 CLIP：

```bash
python zero_shot.py --data-root /path/to/cub_200_2011
```

Linear Probe：

```bash
python train.py --config configs/clip_linear_probe_cub.yaml --data-root /path/to/cub_200_2011
```

LoRA 微调：

```bash
python train.py --config configs/clip_lora_cub.yaml --data-root /path/to/cub_200_2011 --epochs 10 --batch-size 64
```

或使用服务器一键脚本：

```bash
bash scripts/run_lora_server.sh /path/to/cub_200_2011 10 64
```

AdaptFormer 微调：

```bash
python train.py --config configs/clip_adaptformer_cub.yaml --data-root /path/to/cub_200_2011 --epochs 10 --batch-size 64
```

4090/5090 上如果显存充足，可以把 `--batch-size` 提高到 128；如果显存不足，降到 32。

## 评估与推理

```bash
python evaluate.py --checkpoint outputs/clip_vitb32_lora_cub/best_model.pt --data-root /path/to/cub_200_2011
python predict.py --checkpoint outputs/clip_vitb32_lora_cub/best_model.pt --data-root /path/to/cub_200_2011 --image /path/to/image.jpg
```

## 输出文件

每次训练会在 `outputs/<run_name>/` 下保存：

- `best_model.pt`：最优模型 checkpoint
- `summary.json`：配置、训练历史和最优准确率
- `training_curves.png`：训练/验证 loss 与 accuracy 曲线
- `best_confusion_matrix.png`：最优 epoch 的混淆矩阵

生成报告记录：

```bash
python report_helper.py --run-dir outputs/clip_vitb32_lora_cub
```

## 实验报告可写要点

4.4 实验内容：说明 CUB-200-2011 数据集、CLIP 图像/文本双编码器结构、零样本分类、Linear Probe、LoRA 和 AdaptFormer。

4.5 实验步骤：写明数据读取、CLIP 预训练权重加载、零样本 prompt 构造、冻结主干训练分类头、向视觉编码器注入 LoRA/AdaptFormer、训练与评估流程。

4.6 结果与分析：填入 `summary.json` 中的 accuracy，配上 `training_curves.png` 和 `best_confusion_matrix.png`。分析 LoRA/AdaptFormer 相比零样本的提升，以及参数高效微调在显存和训练速度上的优势。
