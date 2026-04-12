import os
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from tqdm import tqdm
from .base_model import BaseDetector


class FasterRCNNDetector(BaseDetector):
    def __init__(self, config):
        super().__init__(config)
        num_classes = config["model"]["num_classes"]

        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
            weights="DEFAULT"
        )
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    def train_one_epoch(self, dataloader, optimizer, device):
        self.model.train()
        self.model.to(device)
        total_loss = 0.0

        for images, targets in tqdm(dataloader, desc="Train", leave=False):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v
                        for k, v in t.items() if k != "image_id"}
                       for t in targets]

            loss_dict = self.model(images, targets)
            loss = sum(loss_dict.values())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        return {"loss": total_loss / len(dataloader)}

    def evaluate(self, dataloader, device):
        self.model.eval()
        self.model.to(device)
        results = []

        with torch.no_grad():
            for images, targets in tqdm(dataloader, desc="Eval", leave=False):
                images = [img.to(device) for img in images]
                outputs = self.model(images)

                for output, target in zip(outputs, targets):
                    results.append({
                        "image_id": target["image_id"],
                        "boxes": output["boxes"].cpu(),
                        "scores": output["scores"].cpu(),
                        "labels": output["labels"].cpu(),
                    })

        return results

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path, map_location="cpu"))
