"""container_dataset dataset."""
import tensorflow_datasets as tfds
import tensorflow as tf
import json
import numpy as np

# TODO(container_dataset): Markdown description  that will appear on the catalog page.
_DESCRIPTION = """
Description
"""

# TODO(container_dataset): BibTeX citation
_CITATION = """
"""

IMG_HEIGHT = 100
IMG_WIDTH = 100
IMG_DEPTH = 3

class ContainerDataset(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for container_dataset dataset."""

  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
  }


  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""  

    # TODO(container_dataset): Specifies the tfds.core.DatasetInfo object
    return tfds.core.DatasetInfo(
        builder=self,
        description=_DESCRIPTION,
        features=tfds.features.FeaturesDict({
            # These are the features of your dataset like images, labels ...
            ## shape=(height, width, num. of channels)
            'image': tfds.features.Image(shape=(IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH), encoding_format="jpeg"),
            'label': tfds.features.ClassLabel(names=['none', 'container_front']),
            ## shape=(16) janky ass fix
            ## 4, 2 didn't work in training because I couldn't get the layer in the right shape again
            ## So I did 2 to the power of 4 and that's 16.
            'points': tfds.features.Tensor(shape=(8,), dtype=tf.float32, encoding='zlib'),
        }),
        # If there's a common (input, target) tuple from the
        # features, specify them here. They'll be used if
        # `as_supervised=True` in `builder.as_dataset`.
        supervised_keys=('image', ('label', 'points')),  # Set to `None` to disable
        homepage='https://dataset-homepage/',
        citation=_CITATION,
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Returns SplitGenerators."""
    # TODO(container_dataset): Downloads the data and defines the splits
    path = dl_manager.download_and_extract('https://github.com/penguix0/container-detection-dataset/archive/refs/heads/main.zip')
    # TODO(container_dataset): Returns the Dict[split names, Iterator[Key, Example]]
    return {
        'train': self._generate_examples(path / 'container-detection-dataset-main' / 'training'),
        'test': self._generate_examples(path / 'container-detection-dataset-main' / 'testing')
    }

  def _generate_examples(self, path):
    """Yields examples."""
    # TODO(container_dataset): Yields (key, example) tuples from the dataset
    for i, (img, json) in enumerate(zip(path.glob('*.jpg'), path.glob('*.json'))):
      yield i,{
        'image': img,
        'label': self._get_labels(json),
        'points': self._get_points(json)
      }

  def _get_objects(self, json_path):
    print (json_path)
    """"Returns the json content of each image as an array"""
    data = dict()
    points = self._get_points(json_path)
    label = self._get_labels(json_path)

    data["points"] = points
    data["label"] = label

    return data

  def _get_labels(self, path):
    """Returns all labels"""
    train_labels = []

    file = open(path) ## Open the file
    json_file = json.load(file) ## Load the json and save as file
    file.close()

    for shape in json_file["shapes"]:
      label = shape["label"] ## Get the label from the json
      return str(label)

  def _get_points(self, path):
    """Returns all polygon points"""

    array = []

    file = open(path) ## Open the file
    json_file = json.load(file) ## Load the json and save as file
    file.close()

    for shape in json_file["shapes"]:
      points = shape["points"] ## Get the label from the json
      for pair in points:
        for value in pair:
          array.append(value)
              
    return array



