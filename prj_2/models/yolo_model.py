import os
import shutil
import tempfile
import torch
import yaml
from tqdm import tqdm
from ultralytics import YOLO
from .base_model import BaseDetector


class YOLODetector(BaseDetector):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        weights = config["model"].get("weights", "yolov8n.pt")
        self.yolo = YOLO(weights)
        self.model = self.yolo.model  # for base class compat
        self._data_yaml_path = None
        self._trained = False

    def _build_data_yaml(self, train_dataset, val_dataset):
        """Write images to temp dirs and create ultralytics data yaml."""
        tmp_dir = tempfile.mkdtemp(prefix="yolo_data_")
        train_img_dir = os.path.join(tmp_dir, "images", "train")
        val_img_dir = os.path.join(tmp_dir, "images", "val")
        train_lbl_dir = os.path.join(tmp_dir, "labels", "train")
        val_lbl_dir = os.path.join(tmp_dir, "labels", "val")
        for d in [train_img_dir, val_img_dir, train_lbl_dir, val_lbl_dir]:
            os.makedirs(d, exist_ok=True)

        class_names = (train_dataset.get_class_names()
                       if hasattr(train_dataset, "get_class_names")
                       else train_dataset.class_names)
        num_classes = len(class_names)

        def write_split(dataset, img_dir, lbl_dir):
            from torchvision.transforms.functional import to_pil_image
            for i in range(len(dataset)):
                img_tensor, target = dataset[i]
                image_id = target["image_id"]
                img_pil = to_pil_image(img_tensor)
                img_path = os.path.join(img_dir, f"{image_id}.jpg")
                img_pil.save(img_path)

                w, h = img_pil.size
                boxes = target["boxes"]
                labels = target["labels"]
                lines = []
                for box, label in zip(boxes.tolist(), labels.tolist()):
                    x1, y1, x2, y2 = box
                    cx = ((x1 + x2) / 2) / w
                    cy = ((y1 + y2) / 2) / h
                    bw = (x2 - x1) / w
                    bh = (y2 - y1) / h
                    cls_idx = int(label) - 1  # YOLO uses 0-indexed
                    lines.append(f"{cls_idx} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
                with open(os.path.join(lbl_dir, f"{image_id}.txt"), "w") as f:
                    f.write("\n".join(lines))

        print("Preparing YOLO train images...")
        write_split(train_dataset, train_img_dir, train_lbl_dir)
        print("Preparing YOLO val images...")
        write_split(val_dataset, val_img_dir, val_lbl_dir)

        data_yaml = {
            "path": tmp_dir,
            "train": "images/train",
            "val": "images/val",
            "nc": num_classes,
            "names": class_names,
        }
        yaml_path = os.path.join(tmp_dir, "data.yaml")
        with open(yaml_path, "w") as f:
            yaml.dump(data_yaml, f)

        self._data_yaml_path = yaml_path
        self._tmp_dir = tmp_dir
        self._class_names = class_names
        return yaml_path

    def train_full(self, train_dataset, val_dataset):
        """YOLO uses its own training loop — call this instead of train_one_epoch."""
        train_cfg = self.config["training"]
        ckpt_dir = os.path.abspath(self.config["output"]["checkpoint_dir"])
        os.makedirs(ckpt_dir, exist_ok=True)

        yaml_path = self._build_data_yaml(train_dataset, val_dataset)

        self.yolo.train(
            data=yaml_path,
            epochs=train_cfg.get("epochs", 5),
            batch=train_cfg.get("batch_size", 16),
            imgsz=self.config["dataset"].get("image_size", 640),
            lr0=train_cfg.get("lr", 0.01),
            device=0 if train_cfg.get("device", "cuda") == "cuda" else "cpu",
            project=ckpt_dir,
            name="run",
            exist_ok=True,
            verbose=False,
        )
        self._trained = True

        # Copy best weights to checkpoint dir root
        best_pt = os.path.join(ckpt_dir, "run", "weights", "best.pt")
        if os.path.exists(best_pt):
            shutil.copy(best_pt, os.path.join(ckpt_dir, "best.pt"))

    def train_one_epoch(self, dataloader, optimizer, device):
        # Not used for YOLO — train_full handles everything
        raise NotImplementedError("Use train_full() for YOLO")

    def evaluate(self, dataloader, device):
        results = []
        class_names = getattr(self, "_class_names", None)

        for images, targets in tqdm(dataloader, desc="Eval YOLO", leave=False):
            for img_tensor, target in zip(images, targets):
                from torchvision.transforms.functional import to_pil_image
                img_pil = to_pil_image(img_tensor)

                preds = self.yolo.predict(img_pil, verbose=False, conf=0.25)[0]
                boxes_xyxy = preds.boxes.xyxy.cpu()
                scores = preds.boxes.conf.cpu()
                cls_ids = preds.boxes.cls.cpu().long()

                # Map YOLO class ids (0-indexed) to 1-indexed labels
                labels = cls_ids + 1

                results.append({
                    "image_id": target["image_id"],
                    "boxes": boxes_xyxy,
                    "scores": scores,
                    "labels": labels,
                })

        return results

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        ckpt_dir = self.config["output"]["checkpoint_dir"]
        best_pt = os.path.join(ckpt_dir, "best.pt")
        if os.path.exists(best_pt):
            shutil.copy(best_pt, path)

    def load(self, path):
        self.yolo = YOLO(path)
        self.model = self.yolo.model
