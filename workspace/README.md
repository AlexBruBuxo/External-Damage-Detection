# Training:

*Source: https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_training_and_evaluation.md*

### Run model_main_tf2.py:

1) cd into 'workspace/' directory:

2) optional: select desired GPUs, if nothing is set, it will use all available GPUs
```
> export CUDA_VISIBLE_DEVICES=0
> export CUDA_VISIBLE_DEVICES=1
> export CUDA_VISIBLE_DEVICES=0,1 --> Both
> export CUDA_VISIBLE_DEVICES=-1 --> None
```

3) run the model:

     a) original commands:
```
python model_main_tf2.py \
  --pipeline_config_path="./models/efficientdet_d0/v1/pipeline.config" \
  --model_dir="./models/efficientdet_d0/v1/" \
  --checkpoint_every_n=200 \
  --num_workers=20 \
  --alsologtostderr
```

     b) nohup commands:
```
nohup python model_main_tf2.py \
  --pipeline_config_path="./models/efficientdet_d0/v2/pipeline.config" \
  --model_dir="./models/efficientdet_d0/v2/" \
  --checkpoint_every_n=200 \
  --num_workers=20 \
  --alsologtostderr > tmp/nohup_efficientdet_d0_v2_1.txt &
```

```
<pipeline_config_path> is a path to the config file you are going to use for the current training job. Should be a config file from ./models/<folder with the model of your choice>/v1/

<model_dir> is a path to a directory where all of your future model attributes will be placed. Should also be the following: ./models/<folder with the model of your choice>/v1/ 

<checkpoint_every_n> is an integer that defines how many steps should be completed in a sequence order to make a model checkpoint. Remember, that when a single step is made, your model processes a number of images equal to your batch_size defined for training.

<num_workers> if you have a multi-core CPU, this parameter defines the number of cores that can be used for the training job.
```

4) Check tensorboard (from the 'workspace' directory):
```
tensorboard --logdir models/
```

___

### Run exporter_main_v2.py:  

1) From 'workspace/' directory:

2) Run the script:
```
python exporter_main_v2.py \
  --pipeline_config_path="./models/efficientdet_d0/v2/pipeline.config" \
  --trained_checkpoint_dir="./models/efficientdet_d0/v2/" \
  --output_directory="./exported_models/efficientdet_d0/" \
  --input_type=image_tensor
```

```
<path to your config file> is a path to the config file for the model you want to export. Should be a config file from ./models/<folder with the model of your choice>/v1/ 

<path to a directory with your trained model> is a path to a directory where model checkpoints were placed during training. Should also be the following: ./models/<folder with the model of your choice>/v1/  

<path to a directory where to export a model> is a path where an exported model will be saved. Should be: ./exported_models/<folder with the model of your choice> 
```

___

### Evaluation

1) cd into 'workspace/' directory:

2) optional: select desired GPUs, if nothing is set, it will use all available GPUs
```
> export CUDA_VISIBLE_DEVICES=0
> export CUDA_VISIBLE_DEVICES=1
> export CUDA_VISIBLE_DEVICES=0,1
```

3) run the evaluation process:

     a) original command:
```
python model_main_tf2.py \
  --pipeline_config_path="./models/efficientdet_d0/v1/pipeline.config" \
  --model_dir="./models/efficientdet_d0/v1/" \
  --checkpoint_every_n=200 \
  --num_workers=20 \
  --alsologtostderr
```

     b) nohup command:
```
nohup python model_main_tf2.py \
  --pipeline_config_path="./models/efficientdet_d0/v2/pipeline.config" \
  --model_dir="./models/efficientdet_d0/v2/" \
  --checkpoint_dir="./models/efficientdet_d0/v2/" \
  --alsologtostderr > tmp/nohup_efficientdet_d0_v2_1_eval.txt &
```
