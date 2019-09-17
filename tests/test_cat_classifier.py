import os
import unittest
import warnings

from ipdb import set_trace

from puzzlesolver.classifiers.cat import CatPredictor
from puzzlesolver.classifiers.util import GenerateTFRecord, TFRecordExtractor
from puzzlesolver.utils import get_project_root

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    import tensorflow as tf
    from tensorflow import keras

PROJECT_ROOT = get_project_root()
TFRECORD_PATH = "train_images.tfrecord"
MODEL_PATH = "models/cat_model.cpt"


class TestCatClassifierModel(unittest.TestCase):
    def setUp(self):
        self.data_folder = os.path.join(PROJECT_ROOT, "data")
        self.model_folder = os.path.join(PROJECT_ROOT, MODEL_PATH)
        # get model
        self.cat_model = CatPredictor()
        self.cat_model.compile(
            optimizer="rmsprop",
            loss=tf.losses.categorical_crossentropy,
            metrics=["accuracy"],
        )
        # get data for training
        self.test_TFRecord_path = os.path.join(self.data_folder, TFRECORD_PATH)
        self.t = TFRecordExtractor(self.test_TFRecord_path)
        self.train_set, self.val_set = self.t.extract_image()

    def test_cat_model_is_keras_model(self):
        self.assertIsInstance(self.cat_model, CatPredictor.__bases__)

    def test_cat_model_trains(self):
        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=self.model_folder, monitor="val_loss", save_best_only=True
        )

        self.cat_model.fit(
            self.train_set,
            epochs=3,
            validation_data=self.val_set,
            callbacks=[checkpoint_callback],
        )
