# 目标检测实验

在 VOC2007 和 COCO2017 上训练并评估 Faster-RCNN、YOLO、SSD、DETR 四个模型。评估工具使用 [rafaelpadilla/Object-Detection-Metrics](https://github.com/rafaelpadilla/Object-Detection-Metrics)。

## 1. 安装依赖

```bash
pip install -r requirements.txt
```

## 2. 下载数据集和评估工具

```bash
bash scripts/download_data.sh
```

## 3. 训练所有模型

建议在 tmux 中运行防止 SSH 断连：

```bash
tmux new -s train
```

```bash
python train.py --config configs/faster_rcnn_voc.yaml
python train.py --config configs/faster_rcnn_coco.yaml
python train.py --config configs/ssd_voc.yaml
python train.py --config configs/ssd_coco.yaml
python train.py --config configs/yolo_voc.yaml
python train.py --config configs/yolo_coco.yaml
```

DETR 需要先设置 HuggingFace 镜像：

```bash
export HF_ENDPOINT=https://hf-mirror.com
python train.py --config configs/detr_voc.yaml
python train.py --config configs/detr_coco.yaml
```

## 4. 跑评估，生成结果文件

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

## 5. 计算 mAP

需要在 `Object-Detection-Metrics` 目录下运行，先创建输出目录，再逐个计算：

```bash
cd Object-Detection-Metrics

mkdir -p ../results/{faster_rcnn_voc,faster_rcnn_coco,ssd_voc,ssd_coco,yolo_voc,yolo_coco,detr_voc,detr_coco}/metrics

python pascalvoc.py -gt ../results/faster_rcnn_voc/groundtruth -det ../results/faster_rcnn_voc/detections -t 0.5 -sp ../results/faster_rcnn_voc/metrics
python pascalvoc.py -gt ../results/faster_rcnn_coco/groundtruth -det ../results/faster_rcnn_coco/detections -t 0.5 -sp ../results/faster_rcnn_coco/metrics
python pascalvoc.py -gt ../results/ssd_voc/groundtruth -det ../results/ssd_voc/detections -t 0.5 -sp ../results/ssd_voc/metrics
python pascalvoc.py -gt ../results/ssd_coco/groundtruth -det ../results/ssd_coco/detections -t 0.5 -sp ../results/ssd_coco/metrics
python pascalvoc.py -gt ../results/yolo_voc/groundtruth -det ../results/yolo_voc/detections -t 0.5 -sp ../results/yolo_voc/metrics
python pascalvoc.py -gt ../results/yolo_coco/groundtruth -det ../results/yolo_coco/detections -t 0.5 -sp ../results/yolo_coco/metrics
python pascalvoc.py -gt ../results/detr_voc/groundtruth -det ../results/detr_voc/detections -t 0.5 -sp ../results/detr_voc/metrics
python pascalvoc.py -gt ../results/detr_coco/groundtruth -det ../results/detr_coco/detections -t 0.5 -sp ../results/detr_coco/metrics
```

每个模型的 AP 图和 mAP 结果保存在对应的 `results/<模型>_<数据集>/metrics/` 下。