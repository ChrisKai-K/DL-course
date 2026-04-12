from abc import ABC, abstractmethod
import torch


class BaseDetector(ABC):
    def __init__(self, config):
        self.config = config
        self.model = None

    @abstractmethod
    def train_one_epoch(self, dataloader, optimizer, device):
        """Returns dict with key 'loss' (float)."""

    @abstractmethod
    def evaluate(self, dataloader, device):
        """
        Returns list of dicts:
            {"image_id": str, "boxes": Tensor Nx4 xyxy, "scores": Tensor N, "labels": Tensor N}
        """

    @abstractmethod
    def save(self, path):
        pass

    @abstractmethod
    def load(self, path):
        pass

    def get_optimizer(self):
        train_cfg = self.config["training"]
        params = [p for p in self.model.parameters() if p.requires_grad]
        return torch.optim.SGD(
            params,
            lr=train_cfg.get("lr", 0.005),
            momentum=train_cfg.get("momentum", 0.9),
            weight_decay=train_cfg.get("weight_decay", 0.0005),
        )
