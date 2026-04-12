from .voc_dataset import VOCDetectionSubset, VOC_CLASSES
from .coco_dataset import COCODetectionSubset


def build_dataset(config, split="train"):
    ds_cfg = config["dataset"]
    name = ds_cfg["name"]
    max_key = "max_train" if split == "train" else "max_val"
    max_images = ds_cfg.get(max_key)
    image_size = ds_cfg.get("image_size", 800)

    if name == "voc":
        return VOCDetectionSubset(
            root=ds_cfg["root"],
            split=split,
            max_images=max_images,
            image_size=image_size,
        )
    elif name == "coco":
        return COCODetectionSubset(
            root=ds_cfg["root"],
            split=split,
            max_images=max_images,
            image_size=image_size,
        )
    else:
        raise ValueError(f"Unknown dataset: {name}")
