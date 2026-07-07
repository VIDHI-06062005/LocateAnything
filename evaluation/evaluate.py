"""
evaluate.py

Runs evaluation of LocateAnything-3B on the COCO validation dataset.
"""

import time
from PIL import Image
from tqdm import tqdm

from config import EVALUATION_CONFIG
from inference import run_inference
from visualization import (
    extract_prediction,
    normalize_to_pixels
)

from evaluation.coco_loader import COCOLoader
from evaluation.matcher import find_best_match
from evaluation.report import (
    save_csv,
    save_summary
)


loader = COCOLoader(
    image_dir="datasets/coco/val2017",
    annotation_file="datasets/coco/annotations/instances_val2017.json"
)


results = []

iou_scores = []

success = 0

failed = 0

inference_times = []

for index in tqdm(range(EVALUATION_CONFIG["num_images"])):

    sample = loader.get_image(index)

    image = Image.open(
        sample["image_path"]
    ).convert("RGB")

    annotations = sample["annotations"]

    if len(annotations) == 0:
        continue

    # ---------------------------------------------
    # Use first object as evaluation query
    # ---------------------------------------------

    first_ann = annotations[0]

    gt_label = loader.get_category_name(
        first_ann["category_id"]
    )

    query = f"Find the {gt_label}"

    # ---------------------------------------------
    # Run model
    # ---------------------------------------------

    start = time.time()

    response = run_inference(
        image,
        query
    )

    inference_time = time.time() - start
    inference_times.append(inference_time)

    predicted_label, predicted_box = extract_prediction(
        response
    )

    if predicted_box is None:

        failed += 1

        continue

    predicted_box = normalize_to_pixels(
        predicted_box,
        sample["width"],
        sample["height"]
    )

    # ---------------------------------------------
    # Match prediction with GT
    # ---------------------------------------------

    best_iou, best_ann = find_best_match(

        predicted_label,

        predicted_box,

        annotations,

        loader

    )

    if best_ann is None:

        failed += 1

        continue

    success += 1

    iou_scores.append(best_iou)

    results.append({
        "image_id": sample["image_id"],

        "query": query,

        "predicted_label": predicted_label,

        "ground_truth": gt_label,

        "iou": round(best_iou,4),

        "inference_time": round(inference_time,3),

        "status": "Correct" if best_iou >= 0.5 else "Incorrect"

    })


# -------------------------------------------------
# Summary
# -------------------------------------------------

average_iou = (
    sum(iou_scores) / len(iou_scores)
    if len(iou_scores) > 0
    else 0
)

average_time = (
    sum(inference_times)/len(inference_times)
    if inference_times else 0
)

accuracy = (
    success/(success+failed)*100
    if (success+failed)>0 else 0
)

summary = {

    "images_evaluated": success+failed,

    "successful_predictions": success,

    "failed_predictions": failed,

    "detection_accuracy": round(accuracy,2),

    "average_iou": round(average_iou,4),

    "average_inference_time": round(average_time,3)

}

save_csv(
    results,
    EVALUATION_CONFIG["csv_output"]
)

save_summary(
    summary,
    EVALUATION_CONFIG["summary_output"]
)

print("\n" + "=" * 60)

print("Evaluation Complete")

print("=" * 60)

print("\n" + "=" * 60)
print("        LocateAnything-3B Evaluation Report")
print("=" * 60)

print(f"Images Evaluated      : {success + failed}")
print(f"Successful            : {success}")
print(f"Failed                : {failed}")
print(f"Detection Accuracy    : {accuracy:.2f}%")
print(f"Average IoU           : {average_iou:.4f}")
print(f"Average Inference     : {average_time:.3f} sec")

print("=" * 60)
print(f"CSV Report            : {EVALUATION_CONFIG['csv_output']}")
print(f"Summary Report        : {EVALUATION_CONFIG['summary_output']}")
print("=" * 60)
print("Evaluation Finished Successfully!")
print("=" * 60)