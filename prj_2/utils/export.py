import os


def write_detection_txts(predictions, results_dir, class_names):
    """
    predictions: list of dicts with keys:
        image_id (str), boxes (Tensor Nx4 xyxy), scores (Tensor N), labels (Tensor N)
    class_names: list of class name strings (1-indexed, index 0 unused)
    """
    det_dir = os.path.join(results_dir, "detections")
    os.makedirs(det_dir, exist_ok=True)

    for pred in predictions:
        image_id = pred["image_id"]
        boxes = pred["boxes"]
        scores = pred["scores"]
        labels = pred["labels"]

        lines = []
        for box, score, label in zip(boxes.tolist(), scores.tolist(), labels.tolist()):
            label_idx = int(label)
            if label_idx < 1 or label_idx > len(class_names):
                continue
            name = class_names[label_idx - 1].replace(" ", "_")
            x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            lines.append(f"{name} {score:.6f} {x1} {y1} {x2} {y2}")

        with open(os.path.join(det_dir, f"{image_id}.txt"), "w") as f:
            f.write("\n".join(lines))


def write_groundtruth_txts(dataset, results_dir, class_names):
    """
    Iterates dataset directly (no model) and writes GT txt files.
    """
    gt_dir = os.path.join(results_dir, "groundtruth")
    os.makedirs(gt_dir, exist_ok=True)

    for i in range(len(dataset)):
        _, target = dataset[i]
        image_id = target["image_id"]
        boxes = target["boxes"]
        labels = target["labels"]

        lines = []
        for box, label in zip(boxes.tolist(), labels.tolist()):
            label_idx = int(label)
            if label_idx < 1 or label_idx > len(class_names):
                continue
            name = class_names[label_idx - 1].replace(" ", "_")
            x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            lines.append(f"{name} {x1} {y1} {x2} {y2}")

        with open(os.path.join(gt_dir, f"{image_id}.txt"), "w") as f:
            f.write("\n".join(lines))
