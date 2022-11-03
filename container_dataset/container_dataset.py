"""container_dataset dataset."""
import tensorflow_datasets as tfds
import tensorflow as tf
import json
import os

# TODO(container_dataset): Markdown description  that will appear on the catalog page.
_DESCRIPTION = """
Description
"""

# TODO(container_dataset): BibTeX citation
_CITATION = """
"""


class ContainerDataset(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for container_dataset dataset."""

  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
  }
  IMG_HEIGHT = 100 # pixels
  IMG_WIDTH = 100 # pixels

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""
    # TODO(container_dataset): Specifies the tfds.core.DatasetInfo object
    return tfds.core.DatasetInfo(
        builder=self,
        description=_DESCRIPTION,
        features=tfds.features.FeaturesDict({
            # These are the features of your dataset like images, labels ...
            ## shape=(height, width, num. of channels)
            'image': tfds.features.Image(shape=(None, None, 3)),
            'objects': tfds.features.Sequence({
                'label': tfds.features.ClassLabel(names=['container_fron']),
                ## shape=(number points in polygon, 2--> x and y value)
                'points': tfds.features.Tensor(shape=(None, 2), dtype=tf.float32),
            }),
        }),
        # If there's a common (input, target) tuple from the
        # features, specify them here. They'll be used if
        # `as_supervised=True` in `builder.as_dataset`.
        supervised_keys=('image', 'objects'),  # Set to `None` to disable
        homepage='https://dataset-homepage/',
        citation=_CITATION,
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Returns SplitGenerators."""
    # TODO(container_dataset): Downloads the data and defines the splits
    path = dl_manager.download_and_extract('https://github.com/penguix0/container-detection-dataset/archive/refs/heads/main.zip')

    # TODO(container_dataset): Returns the Dict[split names, Iterator[Key, Example]]
    return {
        'train': self._generate_examples(path / 'training'),
        'test': self._generate_examples(path / 'testing')
    }

  def _generate_examples(self, path):
    """Yields examples."""
    # TODO(container_dataset): Yields (key, example) tuples from the dataset
    for i, (img, json) in enumerate(path.glob('*.jpg'), path.glob('*.json')):
      yield i,{
        'image': img,
        'objects': self._get_objects(path)
      }

  def _get_objects(self, path, extension):
    """"Returns the json content of each image as an array"""
    data = dict()
    points = self._get_points(path, extension)
    label = self._get_labels(path, extension)

    data["points"] = points
    data["label"] = label

    return data

  def _get_labels(self, path, extension):
    """Returns all labels"""
    train_labels = []
    for file in os.listdir(path):
        name, ext = os.path.splitext(file) ## split the file in a name and extension
        if ext == extension:
            file = open(os.path.join(path, file)) ## Open the file
            file = json.load(file) ## Load the json and save as file
            
            label = file["shapes"][0]["label"] ## Get the label from the json
            train_labels.append(label) ## Add the label to the set

            file.close()

    return train_labels

  def _get_points(self, path, extension):
    """Returns all polygon points"""
    array = []
    for file in os.listdir(path):
        name, ext = os.path.splitext(file) ## split the file in a name and extension
        if not ext == extension:
            continue

        file = open(os.path.join(path, file)) ## Open the file
        file = json.load(file) ## Load the json and save as file
        
        points = file["shapes"][0]["points"] ## Get the label from the json
        points = tfds.features.Tensor(points, shape=(None, 2), dtype=tf.float32)
        array.append(points)

        file.close()

    return array



