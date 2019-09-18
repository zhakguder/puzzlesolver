import os
import pdb
import shutil
import warnings
from configparser import ConfigParser

import cv2
import numpy as np

from puzzlesolver.utils import get_project_root

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    import tensorflow as tf
    from tensorflow.data import Dataset


PROJECT_ROOT = get_project_root()
config = ConfigParser()
CONFIG_FILE = os.path.join(PROJECT_ROOT, "puzzlesolver/classifiers/config.ini")
config.read(CONFIG_FILE)

data_config = config["dataset"]

DATA_FOLDER = data_config["data_folder"]
TRAIN_FOLDER = data_config["train_folder"]
TFRECORD_PATH = data_config["tfrecord"]
IMAGE_WIDTH = int(data_config["image_width"])
IMAGE_HEIGHT = int(data_config["image_height"])

IMAGE_SIZE = (IMAGE_HEIGHT, IMAGE_WIDTH)
IMAGE_NORM_FACTOR = int(data_config["image_norm_factor"])
DATASET_SIZE = int(data_config["dataset_size"])
TRAIN_SIZE = int(float(data_config["train_prop"]) * DATASET_SIZE)
VAL_SIZE = int(float(data_config["val_prop"]) * DATASET_SIZE)
SHUFFLE_BUFFER_SIZE = int(data_config["shuffle_buffer_size"])
BATCH_SIZE = int(data_config["batch_size"])


ABS_TFRECORD_PATH = os.path.join(DATA_FOLDER, TFRECORD_PATH)


def resize_normalize(image):
    if type(image) == np.ndarray:

        image = tf.image.resize(image, size=IMAGE_SIZE)
        image = tf.reshape(image, (-1, IMAGE_HEIGHT, IMAGE_WIDTH, 3))
    else:
        image = tf.image.resize(image, size=IMAGE_SIZE)
    image /= IMAGE_NORM_FACTOR
    return image


class GenerateTFRecord:
    def __init__(self, labels):
        self.labels = labels
        self.data_folder = os.path.join(PROJECT_ROOT, DATA_FOLDER)
        self.train_folder = os.path.join(self.data_folder, TRAIN_FOLDER)
        self.tfrecord_path = ABS_TFRECORD_PATH

    def convert_image_folder(self):
        img_folder = self.train_folder
        tfrecord_file_name = self.tfrecord_path
        # Get all file names of images present in folder
        img_paths = os.listdir(img_folder)
        img_paths = [os.path.abspath(os.path.join(img_folder, i)) for i in img_paths]

        with tf.io.TFRecordWriter(tfrecord_file_name) as writer:
            for img_path in img_paths:
                example = self._convert_image(img_path)
                writer.write(example.SerializeToString())

    def _convert_image(self, img_path):
        label = self._get_label_with_filename(img_path)
        img_shape = cv2.imread(img_path).shape
        filename = os.path.basename(img_path)

        # Read image data in terms of bytes
        with tf.io.gfile.GFile(img_path, "rb") as fid:
            image_data = fid.read()

        example = tf.train.Example(
            features=tf.train.Features(
                feature={
                    "filename": tf.train.Feature(
                        bytes_list=tf.train.BytesList(value=[filename.encode("utf-8")])
                    ),
                    "rows": tf.train.Feature(
                        int64_list=tf.train.Int64List(value=[img_shape[0]])
                    ),
                    "cols": tf.train.Feature(
                        int64_list=tf.train.Int64List(value=[img_shape[1]])
                    ),
                    "channels": tf.train.Feature(
                        int64_list=tf.train.Int64List(value=[img_shape[2]])
                    ),
                    "image": tf.train.Feature(
                        bytes_list=tf.train.BytesList(value=[image_data])
                    ),
                    "label": tf.train.Feature(
                        int64_list=tf.train.Int64List(value=[label])
                    ),
                }
            )
        )
        return example

    def _get_label_with_filename(self, filename):
        basename = os.path.basename(filename).split(".")[0]
        basename = basename.split("_")[0]
        return self.labels[basename]


class TFRecordExtractor:
    def __init__(self, tfrecord_file):
        self.tfrecord_file = os.path.abspath(tfrecord_file)

    def _extract_fn(self, tfrecord):
        # Extract features using the keys set during creation
        features = {
            "filename": tf.io.FixedLenFeature([], tf.string),
            "rows": tf.io.FixedLenFeature([], tf.int64),
            "cols": tf.io.FixedLenFeature([], tf.int64),
            "channels": tf.io.FixedLenFeature([], tf.int64),
            "image": tf.io.FixedLenFeature([], tf.string),
            "label": tf.io.FixedLenFeature([], tf.int64),
        }

        # Extract the data record
        sample = tf.io.parse_single_example(tfrecord, features)

        image = tf.image.decode_jpeg(sample["image"])
        image = resize_normalize(image)
        img_shape = tf.stack([sample["rows"], sample["cols"], sample["channels"]])
        label = sample["label"]
        filename = sample["filename"]
        return [image, label]

    def extract_image(self):

        # Pipeline of dataset
        dataset = tf.data.TFRecordDataset([self.tfrecord_file])
        dataset = (
            dataset.map(self._extract_fn)
            .shuffle(buffer_size=SHUFFLE_BUFFER_SIZE)
            .batch(BATCH_SIZE)
        )

        train_dataset = dataset.take(TRAIN_SIZE)
        remaining_dataset = dataset.skip(TRAIN_SIZE)
        val_dataset = remaining_dataset.take(VAL_SIZE)
        return train_dataset, val_dataset


class _TestImageEmbedPrepper:
    @staticmethod
    def prep_image(img_path):
        image = cv2.imread(img_path)
        image = resize_normalize(image)
        return image
