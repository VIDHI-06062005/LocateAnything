from evaluation.report import save_csv, save_summary


results = [

    {
        "image_id": 1,
        "query": "Find the bottle",
        "predicted_label": "bottle",
        "ground_truth": "bottle",
        "iou": 0.82
    },

    {
        "image_id": 2,
        "query": "Find the chair",
        "predicted_label": "chair",
        "ground_truth": "chair",
        "iou": 0.74
    }

]


summary = {

    "images_evaluated": 2,
    "average_iou": 0.78,
    "successful_predictions": 2,
    "failed_predictions": 0

}


save_csv(
    results,
    "results/evaluation.csv"
)

save_summary(
    summary,
    "results/summary.json"
)

print("Files created successfully!")