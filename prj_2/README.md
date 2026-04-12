# 目标检测实验

在 VOC2007 和 COCO2017 上训练并评估 Faster-RCNN、YOLO、SSD、DETR 四个模型。评估工具使用 [rafaelpadilla/Object-Detection-Metrics](https://github.com/rafaelpadilla/Object-Detection-Metrics)。

## 1. 环境安装

```bash
pip install -r requirements.txt
```

## 2. 数据下载

```bash
bash scripts/download_data.sh
```

VOC2007 下载到 `data/VOCdevkit/`，COCO2017 val 下载到 `data/coco/`。COCO 使用 val2017（5k 张）在代码中划分：前 1000 张作为训练集，接下来 500 张作为验证集，无需下载 18GB 的训练集。

## 3. 训练

每个模型和数据集组合单独跑一条命令：

```bash
python train.py --config configs/faster_rcnn_voc.yaml
python train.py --config configs/faster_rcnn_coco.yaml
python train.py --config configs/ssd_voc.yaml
python train.py --config configs/ssd_coco.yaml
python train.py --config configs/yolo_voc.yaml
python train.py --config configs/yolo_coco.yaml
python train.py --config configs/detr_voc.yaml
python train.py --config configs/detr_coco.yaml
```

训练完成后 checkpoint 保存到 `checkpoints/<模型>_<数据集>/checkpoint.pt`。

## 4. 评估

先跑推理，生成结果文件：

```bash
python evaluate.py --config configs/faster_rcnn_voc.yaml
python evaluate.py --config configs/faster_rcnn_coco.yaml
python evaluate.py --config configs/ssd_voc.yaml
python evaluate.py --config configs/ssd_coco.yaml
python evaluate.py --config configs/yolo_voc.yaml
python evaluate.py --config configs/yolo_coco.yaml
python evaluate.py --config configs/detr_voc.yaml
python evaluate.py --config configs/detr_coco.yaml
```

再用 rafaelpadilla 工具计算 mAP（以 faster_rcnn_voc 为例，其他替换路径即可）：

```bash
python Object-Detection-Metrics/pascalvoc.py \
  -gt ./results/faster_rcnn_voc/groundtruth \
  -det ./results/faster_rcnn_voc/detections \
  -t 0.5
```

## 注意事项

- 所有模型使用预训练权重，在小规模子集上微调 5 个 epoch（DETR 为 3 个），以保证训练速度。
- DETR 首次运行时会从 HuggingFace 下载预训练权重，需要能访问外网。
- 建议在 `tmux` 中运行训练，防止 SSH 断连导致进程中断：

```bash
tmux new -s train   # 新建会话
# 跑训练命令...
# Ctrl+B 然后 D 退出但保持后台运行
tmux attach -t train  # 重新连回来查看进度
```

