## Import libaries
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import os

# Import the dataset
import container_dataset

# Setup logging
import logging
logger = tf.get_logger()
logger.setLevel(logging.ERROR)


## Import the dataset
filename = "main.py"
data_path = str(__file__).replace(filename, "") + "dataset"

## Load the dataset
(train_dataset, val_dataset, test_dataset), metadata = tfds.load("container_dataset", 
    split=["train[:80%]", "train[80%:]", "test"],
    with_info=True)

## Print classes which the program can detect
class_names = metadata.features['objects']["label"].names
print("Class names: {}".format(class_names))

## Print the number of examples in each set
num_train_examples = metadata.splits['train'].num_examples
num_test_examples = metadata.splits['test'].num_examples
print("Number of training examples: {}".format(num_train_examples))
print("Number of test examples:     {}".format(num_test_examples))

IMG_WIDTH = 100 #pixels
IMG_HEIGHT = 100 #pixels
IMG_DEPTH = 3

train_dataset_img = []
train_dataset_objects = []
## Prepare all images
for i in iter(train_dataset):
    img = i["image"]
    train_dataset_img.append(img)
    objects = []
    for feature in i["objects"]:
        objects.append(i["objects"][feature])
    train_dataset_objects.append(objects)

test_dataset_img = []
test_dataset_objects = []
## Prepare all images
for i in iter(test_dataset):
    img = i["image"]
    test_dataset_img.append(img)
    objects = []
    for feature in i["objects"]:
        objects.append(i["objects"][feature])

    test_dataset_objects.append(objects)

val_dataset_img = []
val_dataset_objects = []
## Prepare all images
for i in iter(val_dataset):
    img = i["image"]
    val_dataset_img.append(img)
    objects = []
    for feature in i["objects"]:
        objects.append(i["objects"][feature])
    val_dataset_objects.append(objects)

train_image_generator      = ImageDataGenerator(rescale=1./255)  # Generator for our training data
test_image_generator       = ImageDataGenerator(rescale=1./255)  # Generator for our test data
validation_image_generator = ImageDataGenerator(rescale=1./255)  # Generator for our validation data


BATCH_SIZE = 5 # Number of training examples to process before updating our models variables

print (train_dataset_objects)


train_data_gen = train_image_generator.flow(x=np.asarray(train_dataset_img).astype('float64'),
                                            y=train_dataset_objects,
                                            batch_size=BATCH_SIZE,
                                            shuffle=True)

test_data_gen = validation_image_generator.flow(x=np.asarray(test_dataset_img).astype('float64'),
                                               y=test_dataset_objects,
                                               batch_size=BATCH_SIZE,
                                               shuffle=False)

val_data_gen = validation_image_generator.flow(x=np.asarray(val_dataset_img).astype('float64'),
                                               y=val_dataset_objects,
                                               batch_size=BATCH_SIZE,
                                               shuffle=False)


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
    tf.convert_to_tensor(train_data_gen),
    steps_per_epoch=int(np.ceil(len(train_dataset_img) / float(BATCH_SIZE))),
    epochs=EPOCHS,
)