## Import libaries
import tensorflow as tf

# Helper libraries
import math
import numpy as np
import matplotlib.pyplot as plt

# Setup logging
import logging
logger = tf.get_logger()
logger.setLevel(logging.ERROR)



## Import the dataset
filename = "main.py"
data_path = str(__file__).replace(filename, "") + "dataset"

import pathlib
### Pathlib makes it easier for us to work with directories and files
dataset_dir = pathlib.Path(data_path)

image_count = len(list(dataset_dir.glob('*/*.jpg')))
print ("Total image count: " + str(image_count))

training_image_count = len(list(dataset_dir.glob('training/*.jpg')))
print ("Total training image count: " + str(training_image_count))

testing_image_count = len(list(dataset_dir.glob('testing/*.jpg')))
print ("Total testing image count: " + str(testing_image_count))