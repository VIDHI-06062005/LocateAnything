"""
visualization.py

Visualization utilities for LocateAnything.
"""

import re
import cv2
import numpy as np


def extract_prediction(text):
    """
    Extract predicted label and bounding box.

    Expected format:

    <ref>chair</ref><box><100><200><500><600></box>

    Returns:
        (label, box)
    """

    label_match = re.search(
        r"<ref>(.*?)</ref>",
        text
    )

    box_match = re.search(
        r"<box><(\d+)><(\d+)><(\d+)><(\d+)></box>",
        text
    )

    if label_match is None or box_match is None:
        return None, None

    label = label_match.group(1)

    box = list(
        map(
            int,
            box_match.groups()
        )
    )

    return label, box


def draw_prediction(image, label, box):
    """
    Draw prediction on image.

    Args:
        image : PIL Image
        label : detected label
        box : normalized coordinates (0-1000)

    Returns:
        RGB NumPy image
    """

    image = np.array(image)

    h, w = image.shape[:2]

    x1 = int(box[0] / 1000 * w)
    y1 = int(box[1] / 1000 * h)
    x2 = int(box[2] / 1000 * w)
    y2 = int(box[3] / 1000 * h)

    cv2.rectangle(
        image,
        (x1, y1),
        (x2, y2),
        (0, 255, 0),
        2
    )

    cv2.putText(
        image,
        label,
        (x1, max(20, y1 - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
        cv2.LINE_AA
    )

    return image


def normalize_to_pixels(box, width, height):
    """
    Convert normalized box (0-1000)
    to pixel coordinates.
    """

    return [
        int(box[0] / 1000 * width),
        int(box[1] / 1000 * height),
        int(box[2] / 1000 * width),
        int(box[3] / 1000 * height),
    ]