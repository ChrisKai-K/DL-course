1. 安装依赖
  pip install -r requirements.txt

  2. 下载数据集和评估工具
  bash scripts/download_data.sh

  3. 训练所有模型（建议用 nohup 或 tmux 跑，防止断连中断）
  # 用 tmux 的话
  tmux new -s train

  python train.py --config configs/faster_rcnn_voc.yaml
  python train.py --config configs/faster_rcnn_coco.yaml
  python train.py --config configs/ssd_voc.yaml
  python train.py --config configs/ssd_coco.yaml
  python train.py --config configs/yolo_voc.yaml
  python train.py --config configs/yolo_coco.yaml

  设置 HuggingFace 镜像再跑：export HF_ENDPOINT=https://hf-mirror.com
  python train.py --config configs/detr_voc.yaml
  python train.py --config configs/detr_coco.yaml

  4. 跑评估 + 生成 txt 文件
  python evaluate.py --config configs/faster_rcnn_voc.yaml
  python evaluate.py --config configs/faster_rcnn_coco.yaml
  # ... 其他 6 个同理

  5. 用 rafaelpadilla 工具算 mAP
  python Object-Detection-Metrics/pascalvoc.py \
    -gt ./results/faster_rcnn_voc/groundtruth \
    -det ./results/faster_rcnn_voc/detections \
    -t 0.5
  每个模型+数据集组合都跑一次，换对应的路径就行。