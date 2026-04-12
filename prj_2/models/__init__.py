from .faster_rcnn_model import FasterRCNNDetector
from .ssd_model import SSDDetector
from .yolo_model import YOLODetector
from .detr_model import DETRDetector

MODEL_REGISTRY = {
    "faster_rcnn": FasterRCNNDetector,
    "ssd": SSDDetector,
    "yolo": YOLODetector,
    "detr": DETRDetector,
}


def build_model(config):
    name = config["model"]["name"]
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {name}. Choose from {list(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name](config)
