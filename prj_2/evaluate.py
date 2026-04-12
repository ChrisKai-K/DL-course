import argparse
import os
import yaml
import torch
from torch.utils.data import DataLoader

from datasets import build_dataset
from models import build_model
from utils.export import write_detection_txts, write_groundtruth_txts


def get_class_names(dataset):
    if hasattr(dataset, "get_class_names"):
        return dataset.get_class_names()
    return dataset.class_names


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to yaml config file")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = torch.device(config["training"].get("device", "cuda")
                          if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    val_dataset = build_dataset(config, split="val")
    print(f"Val: {len(val_dataset)} images")

    class_names = get_class_names(val_dataset)
    results_dir = config["output"]["results_dir"]

    # Write ground truth files
    print("Writing ground truth files...")
    write_groundtruth_txts(val_dataset, results_dir, class_names)

    # Load model and run inference
    detector = build_model(config)
    ckpt_path = os.path.join(config["output"]["checkpoint_dir"], "checkpoint.pt")

    model_name = config["model"]["name"]
    if model_name == "yolo":
        # YOLO loads from best.pt
        best_pt = os.path.join(config["output"]["checkpoint_dir"], "best.pt")
        load_path = best_pt if os.path.exists(best_pt) else ckpt_path
    else:
        load_path = ckpt_path

    if not os.path.exists(load_path):
        print(f"No checkpoint found at {load_path}. Run train.py first.")
        return

    print(f"Loading checkpoint from {load_path}")
    detector.load(load_path)

    batch_size = config["training"].get("batch_size", 4)
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config["training"].get("num_workers", 4),
        collate_fn=val_dataset.collate_fn,
    )

    print("Running inference...")
    predictions = detector.evaluate(val_loader, device)

    print("Writing detection files...")
    write_detection_txts(predictions, results_dir, class_names)

    gt_dir = os.path.join(results_dir, "groundtruth")
    det_dir = os.path.join(results_dir, "detections")

    print("\nDone! Run evaluation with:")
    print(f"  python Object-Detection-Metrics/pascalvoc.py \\")
    print(f"    -gt {gt_dir} \\")
    print(f"    -det {det_dir} \\")
    print(f"    -t 0.5")


if __name__ == "__main__":
    main()
