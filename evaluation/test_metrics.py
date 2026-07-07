from evaluation.metrics import calculate_iou


ground_truth = (100, 100, 300, 300)

prediction = (120, 120, 280, 280)

iou = calculate_iou(
    ground_truth,
    prediction
)

print("=" * 50)
print("Ground Truth :", ground_truth)
print("Prediction   :", prediction)
print("IoU          :", round(iou, 4))
print("=" * 50)