from evaluation.coco_loader import COCOLoader

loader = COCOLoader(
    image_dir="datasets/coco/val2017",
    annotation_file="datasets/coco/annotations/instances_val2017.json"
)

sample = loader.get_image(0)

print("=" * 60)

print("Image ID :", sample["image_id"])

print("Image :", sample["image_path"])

print("Width :", sample["width"])

print("Height :", sample["height"])

print("Objects :", len(sample["annotations"]))

print("First Annotation:")

print(sample["annotations"][0])

print("=" * 60)