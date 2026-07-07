"""
utils.py

Common helper functions used throughout the project.
"""

import os
import cv2

from config import OUTPUT_FOLDER


def create_output_folder():
    """
    Create the output folder if it doesn't exist.
    """
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)


def save_image(image, filename="result.jpg"):
    """
    Save the output image.

    Args:
        image: RGB NumPy image
        filename: Name of the output file

    Returns:
        Path to the saved image
    """

    create_output_folder()

    filepath = os.path.join(OUTPUT_FOLDER, filename)

    image_bgr = cv2.cvtColor(
        image,
        cv2.COLOR_RGB2BGR
    )

    # Save at maximum JPEG quality
    cv2.imwrite(
        filepath,
        image_bgr,
        [cv2.IMWRITE_JPEG_QUALITY, 100]
    )

    return filepath


def validate_query(query):
    """
    Validate the user's query.

    Args:
        query (str): User query

    Returns:
        bool
    """

    if query is None:
        return False

    return len(query.strip()) > 0