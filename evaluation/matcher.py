"""
matcher.py

Matches the predicted object with the best ground-truth
annotation using the highest IoU.
"""

from evaluation.metrics import calculate_iou


def coco_to_xyxy(bbox):
    """
    Convert COCO bbox format

    (x, y, width, height)

    into

    (x1, y1, x2, y2)
    """

    x, y, w, h = bbox

    return (
        x,
        y,
        x + w,
        y + h
    )


def find_best_match(
    predicted_label,
    predicted_box,
    annotations,
    loader
):
    """
    Find the ground-truth object having the
    highest IoU.

    Returns

    best_iou
    best_annotation
    """

    best_iou = 0.0
    best_annotation = None

    for ann in annotations:

        gt_label = loader.get_category_name(
            ann["category_id"]
        ).lower()

        # Skip different classes
        if gt_label != predicted_label.lower():
            continue

        gt_box = coco_to_xyxy(
            ann["bbox"]
        )

        iou = calculate_iou(
            predicted_box,
            gt_box
        )

        if iou > best_iou:

            best_iou = iou
            best_annotation = ann

    return best_iou, best_annotation