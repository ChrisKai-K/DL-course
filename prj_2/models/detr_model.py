import os
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import DetrForObjectDetection, DetrImageProcessor
from .base_model import BaseDetector


class DETRDetector(BaseDetector):
    def __init__(self, config):
        super().__init__(config)
        num_classes = config["model"]["num_classes"]
        hf_name = config["model"].get("hf_model_name", "facebook/detr-resnet-50")

        self.processor = DetrImageProcessor.from_pretrained(hf_name)
        self.model = DetrForObjectDetection.from_pretrained(
            hf_name,
            num_labels=num_classes,
            ignore_mismatched_sizes=True,
        )

    def get_optimizer(self):
        train_cfg = self.config["training"]
        return torch.optim.AdamW(
            self.model.parameters(),
            lr=train_cfg.get("lr", 1e-4),
            weight_decay=train_cfg.get("weight_decay", 1e-4),
        )

    def _prepare_targets(self, targets, device):
        """Convert targets to DETR format: boxes in cxcywh normalized."""
        detr_targets = []
        for t in targets:
            boxes = t["boxes"].to(device)  # xyxy absolute
            labels = t["labels"].to(device)

            if boxes.shape[0] == 0:
                detr_targets.append({
                    "class_labels": labels,
                    "boxes": boxes,
                })
                continue

            # Need image size to normalize — use a dummy size if not available
            # We'll normalize after getting actual image dimensions
            detr_targets.append({
                "class_labels": labels,
                "boxes": boxes,  # will be converted below
            })
        return detr_targets

    def train_one_epoch(self, dataloader, optimizer, device):
        self.model.train()
        self.model.to(device)
        total_loss = 0.0

        for images, targets in tqdm(dataloader, desc="Train DETR", leave=False):
            # images: list of tensors C x H x W
            pixel_values_list = []
            detr_targets = []

            for img_tensor, target in zip(images, targets):
                h, w = img_tensor.shape[1], img_tensor.shape[2]
                boxes_xyxy = target["boxes"]  # Nx4 xyxy absolute
                labels = target["labels"]

                # Convert xyxy absolute -> cxcywh normalized
                if boxes_xyxy.shape[0] > 0:
                    cx = (boxes_xyxy[:, 0] + boxes_xyxy[:, 2]) / 2 / w
                    cy = (boxes_xyxy[:, 1] + boxes_xyxy[:, 3]) / 2 / h
                    bw = (boxes_xyxy[:, 2] - boxes_xyxy[:, 0]) / w
                    bh = (boxes_xyxy[:, 3] - boxes_xyxy[:, 1]) / h
                    boxes_norm = torch.stack([cx, cy, bw, bh], dim=1)
                else:
                    boxes_norm = torch.zeros((0, 4))

                pixel_values_list.append(img_tensor)
                detr_targets.append({
                    "class_labels": labels.to(device),
                    "boxes": boxes_norm.to(device),
                })

            # Pad images to same size
            pixel_values = torch.stack(pixel_values_list).to(device)

            outputs = self.model(pixel_values=pixel_values, labels=detr_targets)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.1)
            optimizer.step()
            total_loss += loss.item()

        return {"loss": total_loss / len(dataloader)}

    def evaluate(self, dataloader, device):
        self.model.eval()
        self.model.to(device)
        results = []

        with torch.no_grad():
            for images, targets in tqdm(dataloader, desc="Eval DETR", leave=False):
                for img_tensor, target in zip(images, targets):
                    h, w = img_tensor.shape[1], img_tensor.shape[2]
                    pixel_values = img_tensor.unsqueeze(0).to(device)

                    outputs = self.model(pixel_values=pixel_values)

                    # post_process returns xyxy absolute boxes
                    target_sizes = torch.tensor([[h, w]])
                    processed = self.processor.post_process_object_detection(
                        outputs, threshold=0.5, target_sizes=target_sizes
                    )[0]

                    results.append({
                        "image_id": target["image_id"],
                        "boxes": processed["boxes"].cpu(),
                        "scores": processed["scores"].cpu(),
                        "labels": processed["labels"].cpu(),
                    })

        return results

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path, map_location="cpu"))
