## Import libaries
import tensorflow as tf


# Helper libraries
import math
import numpy as np
import matplotlib.pyplot as plt
import os
import json

# Setup logging
import logging
logger = tf.get_logger()
logger.setLevel(logging.ERROR)


## Import the dataset
filename = "main.py"
data_path = str(__file__).replace(filename, "") + "dataset"

base_dir = os.path.join(data_path)
train_dir= os.path.join(base_dir, "training")
validation_dir = os.path.join(base_dir, "testing")

## Count images
training_image_count = 0
for file in os.listdir(train_dir):
    name, ext = os.path.splitext(file)
    if ext == '.json':
        training_image_count += 1
print ("Total training image count: " + str(training_image_count))

validation_image_count = 0
for file in os.listdir(validation_dir):
    name, ext = os.path.splitext(file)
    if ext == '.json':
        validation_image_count += 1
print ("Total testing image count: " + str(validation_image_count))


## Get all labels from the images

def get_labels(dir, extension):
    train_labels = set() ## Using a set to make sure not duplicates are added
    for file in os.listdir(dir):
        name, ext = os.path.splitext(file) ## split the file in a name and extension
        if ext == extension:
            file = open(os.path.join(train_dir, file)) ## Open the file
            file = json.load(file) ## Load the json and save as file
            label = file["shapes"][0]["label"] ## Get the label from the json
            train_labels.add(label) ## Add the label to the set

    return train_labels

training_labels = get_labels(train_dir, '.json')
print (training_labels)

## Image dimensions in pixels
IMG_WIDTH = 100
IMG_HEIGHT = 100
### 3 layers: red, green, blue
IMG_DEPTH = 3

BATCH_SIZE = 5# Number of training examples to process before updating our models variables

def convert_img_to_array(image_file):
    img = tf.keras.preprocessing.image.load_img(image_file)

    image_array = tf.keras.preprocessing.image.img_to_array(img)
    print(image_array.shape)
    print(image_array)

convert_img_to_array(os.path.join(train_dir, "IMG_010.jpg"))