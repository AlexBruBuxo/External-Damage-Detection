"""
Second script

Before executing this file, 'create_empty_annotations.py' is executed.
This script is used to create the general COCO json annotations from the original YOLO annotations.
After this, the 'coco_split.py' script is used to split between train/val/test.
"""


from globox import AnnotationSet
from pathlib import Path

cwd = Path().resolve()

yolo_path_files = cwd / "yolo_dataset/NordTank586x371/labels/"
yolo_path_images = cwd / "yolo_dataset/NordTank586x371/images/"
coco_path = cwd / "coco_labels"

# read annotations from YOLO
gts = AnnotationSet.from_yolo(
    folder=yolo_path_files,
    image_folder=yolo_path_images,
    image_extension = ".png")

# Show current stats
gts.show_stats()

# create COCO json annotations file 
gts.save_coco(
    coco_path,
    label_to_id = {"dirt": 0, "damage": 1},
    auto_ids=True)

print('DONE: Saved as COCO annotations.')