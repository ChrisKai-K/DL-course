#!/bin/bash
set -e

mkdir -p data/VOCdevkit
mkdir -p data/coco/images
mkdir -p data/coco/annotations

echo "=== Downloading VOC2007 ==="
wget -c http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar -P data/
tar -xf data/VOCtrainval_06-Nov-2007.tar -C data/
wget -c http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar -P data/
tar -xf data/VOCtest_06-Nov-2007.tar -C data/
echo "VOC2007 done."

echo "=== Downloading COCO2017 val (used as both train/val subset) ==="
wget -c http://images.cocodataset.org/zips/val2017.zip -P data/coco/
unzip -q data/coco/val2017.zip -d data/coco/images/
wget -c http://images.cocodataset.org/annotations/annotations_trainval2017.zip -P data/coco/
unzip -q data/coco/annotations_trainval2017.zip -d data/coco/
echo "COCO2017 done."

echo "=== Cloning Object-Detection-Metrics ==="
if [ ! -d "Object-Detection-Metrics" ]; then
    git clone https://github.com/rafaelpadilla/Object-Detection-Metrics.git
fi
echo "All done."
