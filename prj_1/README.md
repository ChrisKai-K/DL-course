# Project 1: Image Classification

This experiment follows the classification handout and focuses on:

- CIFAR-10 classification
- CUB-200-2011 fine-grained bird classification
- model comparison across AlexNet, ResNet, SENet, and ViT
- evaluation with accuracy, recall, and confusion matrix

## Directory Layout

- `configs/`: experiment configs
- `notebooks/`: optional exploration notebooks
- `reports/`: figures, tables, and analysis notes
- `scripts/`: helper commands for training or evaluation
- `src/datasets/`: dataset loading and transforms
- `src/models/`: model definitions or wrappers
- `src/utils/`: metrics and misc helpers
- `train.py`: training entrypoint
- `evaluate.py`: evaluation entrypoint

## Covered Tasks

The project now covers the main requirements in the handout:

- CIFAR-10 classification
- CUB-200-2011 classification
- model interfaces for AlexNet, ResNet, SENet, and ViT
- accuracy, macro recall, and confusion matrix
- training curves and checkpoint saving

## Environment

Install dependencies:

```bash
pip install -r ../requirements.txt
```

## Data Preparation

### CIFAR-10

No manual download is required. The dataset is downloaded automatically by `torchvision`.

### CUB-200-2011

Extract the dataset to a directory like:

```text
data/cub_200_2011/CUB_200_2011/
  images/
  images.txt
  image_class_labels.txt
  train_test_split.txt
```

If your dataset is stored elsewhere, pass `--data-root /path/to/cub_root`.

## Training

Train a CIFAR-10 ResNet18 baseline:

```bash
cd prj_1
python train.py --config configs/cifar10_resnet18.yaml
```

Train a CIFAR-10 SENet baseline:

```bash
python train.py --config configs/cifar10_senet.yaml
```

Train on CUB-200-2011:

```bash
python train.py --config configs/cub_resnet50.yaml --data-root ../data/cub_200_2011
```

## Evaluation

```bash
python evaluate.py \
  --config configs/cifar10_resnet18.yaml \
  --checkpoint outputs/cifar10_resnet18_baseline/best_model.pt
```

## Outputs

Each run writes:

- `best_model.pt`
- `best_confusion_matrix.png`
- `training_curves.png`
- `summary.json`

These files are suitable for your report figures and experiment analysis.
