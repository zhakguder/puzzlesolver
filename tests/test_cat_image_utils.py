import os
import unittest
from configparser import ConfigParser

import numpy as np

from puzzlesolver.classifiers.util import (GenerateTFRecord, TFRecordExtractor,
                                           _TestImageEmbedPrepper)
from puzzlesolver.utils import get_project_root

config = ConfigParser()
ROOT = get_project_root()
config_path = os.path.join(ROOT, "puzzlesolver", "classifiers", "config.ini")
config.read(config_path)
data_config = config["dataset"]


class TestCatClassifierUtils(unittest.TestCase):
    def setUp(self):
        self.g = GenerateTFRecord({"cat": 0, "dog": 1})
        self.g.convert_image_folder()
        self.test_TFRecord_path = os.path.join(ROOT, "data", "train_images.tfrecord")

        self.t = TFRecordExtractor(self.test_TFRecord_path)
        self.train_set, self.val_set = self.t.extract_image()
        self.batch_size = int(data_config["batch_size"])

    def test_can_read_data(self):
        self.assertTrue(os.path.isfile(self.test_TFRecord_path))

    def test_can_extract_single_batch(self):

        for imgs, labels in self.train_set.take(1):
            self.assertEqual(
                len(labels), self.batch_size, "can't take batch number of examples"
            )

    def test_can_normalize_image(self):

        for imgs, labels in self.train_set.take(1):
            self.assertTrue(np.all(imgs <= 1.0), "image normalization failed")
            self.assertTrue(np.all(imgs >= 0.0), "image normalization failed")

    def test_can_split_dataset(self):
        for imgs, labels in self.val_set.take(1):
            self.assertTrue(np.all(imgs <= 1.0), "image normalization failed")
            self.assertTrue(np.all(imgs >= 0.0), "image normalization failed")
