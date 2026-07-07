"""
coco_loader.py

Loads the COCO 2017 validation dataset.
"""

import os
from pycocotools.coco import COCO


class COCOLoader:
    """
    Helper class for loading COCO validation images and annotations.
    """

    def __init__(self, image_dir, annotation_file):

        self.image_dir = image_dir
        self.coco = COCO(annotation_file)

        self.image_ids = self.coco.getImgIds()

        categories = self.coco.loadCats(
            self.coco.getCatIds()
        )

        self.category_map = {
            cat["id"]: cat["name"]
            for cat in categories
        }

    def __len__(self):
        return len(self.image_ids)

    def get_image(self, index):

        image_id = self.image_ids[index]

        image_info = self.coco.loadImgs(image_id)[0]

        image_path = os.path.join(
            self.image_dir,
            image_info["file_name"]
        )

        ann_ids = self.coco.getAnnIds(imgIds=image_id)

        anns = self.coco.loadAnns(ann_ids)

        return {
            "image_id": image_id,
            "image_path": image_path,
            "width": image_info["width"],
            "height": image_info["height"],
            "annotations": anns
        }

    def get_category_name(self, category_id):

        return self.category_map.get(
            category_id,
            "Unknown"
        )