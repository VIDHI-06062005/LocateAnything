"""
report.py

Utility functions for saving evaluation results.
"""

import os
import csv
import json


def save_csv(results, output_file):
    """
    Save evaluation results to CSV.
    """

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, "w", newline="") as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "image_id",
                "query",
                "predicted_label",
                "ground_truth",
                "iou",
                "inference_time",
                "status"
            ]
            
        )

        writer.writeheader()

        writer.writerows(results)


def save_summary(summary, output_file):
    """
    Save summary statistics to JSON.
    """

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, "w") as file:

        json.dump(
            summary,
            file,
            indent=4
        )