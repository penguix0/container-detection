## Import libaries
from tkinter import N
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


## Get all labels from the images and count training point
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

## Prepare all images for training

def convert_img_to_array(image_file):
    img = tf.keras.preprocessing.image.load_img(image_file)
    image_array = tf.keras.preprocessing.image.img_to_array(img)
    return image_array

def convert_all_images_to_one_array(dir, extension):
    ##array = np.array([])
    array = []
    for file in os.listdir(dir):
        name, ext = os.path.splitext(file) ## split the file in a name and extension
        if not ext == extension:
            continue
        img_array = convert_img_to_array(os.path.join(dir, file))
        array.append(img_array)
        ##np.concatenate(array, img_array)

    print ("Conversion of images in " + str(dir) + " succesfull")
    return array
        
##training_image_data = convert_all_images_to_one_array(train_dir, ".jpg")
training_image_data = tf.keras.utils.image_dataset_from_directory(
    directory=train_dir,
    labels=None,
    class_names=None,
    color_mode="rgb",
    batch_size=BATCH_SIZE,
    image_size=(IMG_HEIGHT, IMG_WIDTH), ##Width and height are flipped for some strange reason
    shuffle=True
)

# for batch in training_image_data:
#     print (batch.shape)
#     for image in batch:
#         print (image.shape)

## Prepare all points for training
def get_points(dir, extension):
    array = []
    for file in os.listdir(dir):
        name, ext = os.path.splitext(file) ## split the file in a name and extension
        if not ext == extension:
            continue

        file = open(os.path.join(dir, file)) ## Open the file
        file = json.load(file) ## Load the json and save as file
        
        label = file["shapes"][0]["points"] ## Get the label from the json
        array.append(label)

    return array

training_point_data = get_points(train_dir, ".json")

print (len(training_image_data))
print (len(training_point_data))

# if len(training_image_data) != len(training_point_data):
#     print ("Images don't match, point count, quitting!")
#     quit()

validation_image_data = tf.keras.utils.image_dataset_from_directory(
    directory=validation_dir,
    labels=None,
    class_names=None,
    color_mode="rgb",
    batch_size=BATCH_SIZE,
    image_size=(IMG_HEIGHT, IMG_WIDTH), ##Width and height are flipped for some strange reason
    shuffle=True
)

validation_point_data = get_points(validation_dir, ".json")

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(IMG_WIDTH, IMG_HEIGHT, IMG_DEPTH)),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(4)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

EPOCHS = 14
history = model.fit(
    x=training_image_data,
    y=training_point_data,
    steps_per_epoch=int(np.ceil(len(training_image_data) / float(BATCH_SIZE))),
    epochs=EPOCHS,
    ##validation_split=0.2
    validation_data=(validation_image_data),
)