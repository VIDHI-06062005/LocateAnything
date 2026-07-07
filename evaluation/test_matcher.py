from evaluation.matcher import find_best_match


class DummyLoader:

    def get_category_name(self, category_id):

        mapping = {
            1: "bottle",
            2: "chair"
        }

        return mapping[category_id]


annotations = [

    {
        "category_id": 1,
        "bbox": [100, 100, 200, 200]
    },

    {
        "category_id": 1,
        "bbox": [400, 400, 100, 100]
    },

    {
        "category_id": 2,
        "bbox": [50, 50, 80, 80]
    }

]


predicted_box = (
    120,
    120,
    280,
    280
)


best_iou, ann = find_best_match(
    "bottle",
    predicted_box,
    annotations,
    DummyLoader()
)

print("=" * 50)
print("Best IoU :", round(best_iou, 3))
print("Matched Annotation :")
print(ann)
print("=" * 50)