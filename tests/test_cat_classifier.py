import os
import unittest
import warnings
from configparser import ConfigParser

from puzzlesolver.classifiers.callbacks import (checkpoint_callback,
                                                checkpoint_config)
from puzzlesolver.classifiers.cat import CatPredictor
from puzzlesolver.classifiers.util import (GenerateTFRecord, TFRecordExtractor,
                                           _TestImageEmbedPrepper)
from puzzlesolver.utils import get_project_root

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    import tensorflow as tf
    from tensorflow import keras


PROJECT_ROOT = get_project_root()
TFRECORD_PATH = "train_images.tfrecord"
CONFIG_FILE = os.path.join(PROJECT_ROOT, "puzzlesolver/classifiers/config.ini")
config = ConfigParser()
config.read(CONFIG_FILE)
checkpoint_config = config["checkpoint"]
# MODEL_PATH = "models/cat_model.cpt"
MODEL_PATH = checkpoint_config["weight_path"]


class TestCatClassifierModel(unittest.TestCase):
    def setUp(self):
        self.data_folder = os.path.join(PROJECT_ROOT, "data")
        self.weight_path = os.path.join(PROJECT_ROOT, MODEL_PATH)
        # get model
        self.cat_model = CatPredictor()
        self.cat_model.compile(
            optimizer="rmsprop",
            # loss=tf.losses.categorical_crossentropy,
            loss=tf.nn.softmax_cross_entropy_with_logits,
            metrics=["accuracy"],
        )
        # get data for training
        self.test_TFRecord_path = os.path.join(self.data_folder, TFRECORD_PATH)
        self.t = TFRecordExtractor(self.test_TFRecord_path)
        self.train_set, self.val_set = self.t.extract_image()
        self.image_path = os.path.join(PROJECT_ROOT, "data", "train", "cat.575.jpg")

    @unittest.skip
    def load_model(self):
        return CatPredictor.load_model(self.weight_path)

    @unittest.skip
    def test_cat_model_is_keras_model(self):
        self.assertIsInstance(self.cat_model, CatPredictor.__bases__)

    @unittest.skip
    def test_cat_model_trains(self):
        self.cat_model.fit(
            self.train_set,
            epochs=10,
            validation_data=self.val_set,
            callbacks=[checkpoint_callback],
        )

    @unittest.skip
    def test_can_load_model(self):
        model = self.load_model()
        warnings.warn("Model loaded")

    def test_can_predict(self):
        preds = CatPredictor.predict_cat(self.val_set)
        print(preds)
        self.assertIsNotNone(preds)
        print(preds.shape)

    @unittest.skip
    def test_can_read_test_image(self):
        """Test that a file from disk can be read and a numpy array returned"""
        _TestImageEmbedPrepper.prep_image(self.image_path)

    def test_can_embed_image(self):
        embedding = CatPredictor.embed_image(self.image_path)
        print(embedding)
