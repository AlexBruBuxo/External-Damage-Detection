"""
First script

This script is used to create the non-existent .txt annotation file from YOLO necessary for the 'yolo_to_coco.py'
script to include all images (and no just the ones with bounding boxes).
"""

import os
from pathlib import Path

cwd = Path().resolve()
yolo_path_files = cwd / "yolo_dataset/NordTank586x371/labels/"
yolo_path_images = cwd / "yolo_dataset/NordTank586x371/images/"

# create list with all labels
labels = os.listdir(yolo_path_files)

# remove extension from the label strings
ids = [os.path.splitext(each)[0] for each in labels]

# loop through images and create an empty file if it does not exist
for image in os.listdir(yolo_path_images):
    image_name = os.path.splitext(image)[0]
    if image_name not in ids:
        file = yolo_path_files / (image_name + '.txt')
        # we create a new file, in case it already exists, we use append mode
        # to not overwrite its content
        open(file, 'a').close()
