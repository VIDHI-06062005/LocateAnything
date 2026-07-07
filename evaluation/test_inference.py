from PIL import Image

from inference import run_inference

image = Image.open(
    "datasets/coco/val2017/000000000139.jpg"
).convert("RGB")

response = run_inference(
    image,
    "Find the bottle"
)

print(type(response))
print(response)