"""
Fourth script

Before executing this file, 'coco_split_annotations.py' is executed.
This script is used to copy the images from the yolo_dataset directory into the desired folders (i.e., train, test, val), based on the split defined in the json files from the coco dataset.

Example usage: 
    python coco_split_images.py coco_dataset/train.json \
      yolo_dataset/NordTank586x371/images coco_dataset/images/train

positional arguments:
  coco_annotations      Path to COCO annotations file.
  src_folder            Where the general dataset is stored.
  dest_folder           Where to store COCO images.

optional arguments:
  -h, --help            show this help message and exit

"""

import json
import argparse
import shutil
import os


def main(args):
    with open(args.annotations, 'rt', encoding='UTF-8') as annotations:
        coco = json.load(annotations)
        info = coco['info']
        licenses = coco['licenses']
        images = coco['images']
        annotations = coco['annotations']
        categories = coco['categories']
        
        number_of_images = len(images)
        
        for image in images:
            src_image_path = os.path.join(args.src, image['file_name'])
            shutil.copy(src_image_path, args.dest)
        
        print(f'Copied {number_of_images} images to {args.dest}')

        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Copies images into desired folder based on COCO annotations.')
    parser.add_argument('annotations', metavar='coco_annotations', type=str,
                        help='Path to COCO annotations file.')
    parser.add_argument('src', metavar='src_folder', type=str, help=' Where the general dataset is stored.')
    parser.add_argument('dest', metavar='dest_folder', type=str, help='Where to store COCO images.')
    args = parser.parse_args()
    
    main(args)