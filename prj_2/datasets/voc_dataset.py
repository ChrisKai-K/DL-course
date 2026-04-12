import random
import torch
from torch.utils.data import Dataset
from torchvision.datasets import VOCDetection
from torchvision import transforms as T

VOC_CLASSES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor",
]
CLASS_TO_IDX = {c: i + 1 for i, c in enumerate(VOC_CLASSES)}  # 1-indexed, 0 = background


class VOCDetectionSubset(Dataset):
    def __init__(self, root, split="train", max_images=None, image_size=800):
        # VOCDetection uses image_set: 'train', 'val', 'trainval', 'test'
        image_set = split if split != "train" else "trainval"
        self.voc = VOCDetection(root=root, year="2007", image_set=image_set, download=False)
        self.image_size = image_size

        indices = list(range(len(self.voc)))
        random.Random(42).shuffle(indices)
        if max_images is not None:
            indices = indices[:max_images]
        self.indices = indices

        self.transform = T.Compose([
            T.ToTensor(),
        ])

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        img, annotation = self.voc[real_idx]

        img_tensor = self.transform(img)

        # Parse annotation
        objs = annotation["annotation"]["object"]
        if not isinstance(objs, list):
            objs = [objs]

        boxes = []
        labels = []
        for obj in objs:
            name = obj["name"]
            if name not in CLASS_TO_IDX:
                continue
            bb = obj["bndbox"]
            x1 = float(bb["xmin"])
            y1 = float(bb["ymin"])
            x2 = float(bb["xmax"])
            y2 = float(bb["ymax"])
            if x2 > x1 and y2 > y1:
                boxes.append([x1, y1, x2, y2])
                labels.append(CLASS_TO_IDX[name])

        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)

        # image_id: VOC filename stem
        image_id = annotation["annotation"]["filename"].replace(".jpg", "")

        target = {"boxes": boxes, "labels": labels, "image_id": image_id}
        return img_tensor, target

    @staticmethod
    def collate_fn(batch):
        images = [item[0] for item in batch]
        targets = [item[1] for item in batch]
        return images, targets

    @staticmethod
    def get_class_names():
        return VOC_CLASSES
