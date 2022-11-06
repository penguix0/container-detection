import pathlib
import numpy
from PIL import Image
img_path = pathlib.Path("/Users/arnecuperus/development/container-detection-dataset/training")
for image in img_path.glob("*.jpg"):
    img = Image.open(image)
    print(image, " ", numpy.asarray(img).shape)

img_path = pathlib.Path("/Users/arnecuperus/development/container-detection-dataset/testing")
for image in img_path.glob("*.jpg"):
    img = Image.open(image)
    print(image, " ", numpy.asarray(img).shape)
