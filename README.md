# Multi-label-Inception
Modified `retrain.py` script to allow multi-label image classification using pretrained [Inception net](https://github.com/tensorflow/models/tree/master/inception).

The `label_image.py` has also been slightly modified to write out the resulting classes and colors into `results.txt`. 

### Requirements
Windows with matlab

python3.5 and [TensorFlow 0.12.0-rc1](https://github.com/tensorflow/tensorflow/releases/tag/0.12.0-rc1)

All the training images must be in JPEG format.

### Run testing datas

#### Prepare training images
1. All the training images should be resized to [112, 92] with the matlab script `myreshape.m` or other methods.

   Put all the training images into **one** folder named `multi-label` inside `images` directory. 
   e.g.  `.\images\multi-label`

#### Prepare labels for each training image
1. We need to prepare files with correct labels for each image.
   Name the files `<image_file_name.jpg>.txt` = if you have an image `car.jpg` the accompanying file will be `car.jpg.txt`. 

   Put each true label on a new line inside the file, nothing else.

   Now copy all the created files into the `image_labels_dir` directory located in project root.
   You can change the path to this folder by editing global variable IMAGE_LABELS_DIR in `retrain.py`

2. Create file `labels.txt` in project root and fill it with all the possible labels. 
   Each label on a new line, nothing else.
   Just like an `image_label` file for an image that is in all the possible classes.

#### Retraining the model
bottleneck_dir will store bottleneck feature outputs
model_dir is where the pretrained model downloaded

python retrain.py --bottleneck_dir=C:\Users\wu\Desktop\zte\Multi-label-Inception\bottlenecks --how_many_training_steps 1000 --model_dir=C:\Users\wu\Desktop\zte\Multi-label-Inception\pretrain-model --output_graph=retrained_graph.pb --output_labels=retrained_labels.txt --summaries_dir=retrain_logs --image_dir=images

#### Testing resulting model
Store all the testing images into one folder named `test`, and don't forget to **resize the test image into [112, 92]!**
Run: `python multi_image.py <folder_name>` from project root.

Run: `python single_image.py <image_path>` from project root.
