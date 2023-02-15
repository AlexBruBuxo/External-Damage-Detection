# Migration of datatset (from YOLO to TFRecords)

This directory includes multiple scripts for the migration of the original dataset in YOLO annotations format to the desired TFRecord format.\
For this, an intermediate step is required: migrating it first to COCO.

## 1. Create empty annotations

The original dataset only has annotations for the images with at least one bounding box. For this, and to also include images with no objects, we can execute the `create_empty_annotations.py` script.

## 2. YOLO to COCO

The second step is to translate the YOLO annotations into the COCO format through `yolo_to_coco.py`. This script creates a general COCO json file.

## 3. Split annotations (train, val, test)

We can then split the general json file into train (70%), validation (20%), and test (10%) sets with `coco_split_annotations.py`. For this, we can choose whether to keep the empty images (i.e., images without any bounding box) or not with the argument `--having-annotations`, which ignores all images without annotations.

For the split, since the script only allows for 2 output files, we can first split between train (70%) and val_test (30%), and then split again the val_test into val (60%), and test (40%); resulting in a split: **train (70%), validation (18%), and test (12%)**:

## 4. Split images (train, val, test)

Once the annotations are split, we can proceed to copy the images from the YOLO directory into the corresponding COCO directories: train, test, val. This is done through the `coco_split_images.py`.

- Train: 2588 images
- Validation: 1056 images
- Test: 812 images

Note that the resultant amount of images do not follow the distribution of 70-18-12 from step 4, as this is counting the overall number of bounding boxes. The images are a result of this.

## 5. COCO to TFRecords

For this final step, we copied the script from the TF Object Detection API `models/research/object_detection/dataset_tools/` into our working environment `create_coco_tf_record.py`.