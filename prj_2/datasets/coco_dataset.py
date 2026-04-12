import os
import random
import torch
from torch.utils.data import Dataset
from torchvision.datasets import CocoDetection
from torchvision import transforms as T


class COCODetectionSubset(Dataset):
    """
    Uses val2017 as the source for both train and val splits.
    train split: first max_train images
    val split:   next max_val images (after max_train)
    This avoids downloading the 18GB train2017.
    """

    def __init__(self, root, split="train", max_images=None, image_size=800):
        img_dir = os.path.join(root, "images", "val2017")
        ann_file = os.path.join(root, "annotations", "instances_val2017.json")
        self.coco_ds = CocoDetection(root=img_dir, annFile=ann_file)
        self.image_size = image_size

        # Build contiguous category id mapping
        coco = self.coco_ds.coco
        cat_ids = sorted(coco.getCatIds())
        self.cat_id_to_label = {cat_id: i + 1 for i, cat_id in enumerate(cat_ids)}
        self.label_to_name = {i + 1: coco.loadCats(cat_id)[0]["name"]
                              for i, cat_id in enumerate(cat_ids)}
        self.class_names = [coco.loadCats(cat_id)[0]["name"] for cat_id in cat_ids]

        all_ids = list(range(len(self.coco_ds)))
        random.Random(42).shuffle(all_ids)

        if split == "train":
            n = max_images if max_images else 1000
            self.indices = all_ids[:n]
        else:
            offset = 1000  # skip train portion
            n = max_images if max_images else 500
            self.indices = all_ids[offset:offset + n]

        self.transform = T.Compose([T.ToTensor()])

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        img, annotations = self.coco_ds[real_idx]
        img_tensor = self.transform(img)

        boxes = []
        labels = []
        for ann in annotations:
            cat_id = ann["category_id"]
            if cat_id not in self.cat_id_to_label:
                continue
            x, y, w, h = ann["bbox"]
            x2, y2 = x + w, y + h
            if w > 0 and h > 0:
                boxes.append([x, y, x2, y2])
                labels.append(self.cat_id_to_label[cat_id])

        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)

        # image_id: zero-padded COCO image id
        img_info = self.coco_ds.coco.loadImgs(self.coco_ds.ids[real_idx])[0]
        image_id = str(img_info["id"]).zfill(12)

        target = {"boxes": boxes, "labels": labels, "image_id": image_id}
        return img_tensor, target

    @staticmethod
    def collate_fn(batch):
        images = [item[0] for item in batch]
        targets = [item[1] for item in batch]
        return images, targets

    def get_class_names(self):
        return self.class_names
