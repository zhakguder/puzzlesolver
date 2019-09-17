import os
import unittest
import warnings

from ipdb import set_trace

from puzzlesolver.classifiers.callbacks import WeightSavingCallback
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
        self.weight_path = os.path.join(PROJECT_ROOT, MODEL_PATH)
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

    def load_model(self):
        return CatPredictor.load_model(self.weight_path)

    def test_cat_model_is_keras_model(self):
        self.assertIsInstance(self.cat_model, CatPredictor.__bases__)

    @unittest.skip
    def test_cat_model_trains(self):
        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=self.weight_path,
            monitor="accuracy",
            save_best_only=True,
            save_weights_only=True,
            save_freq=50,
        )
        weight_cb = WeightSavingCallback(self.weight_path)
        self.cat_model.fit(
            self.train_set,
            epochs=1,
            validation_data=self.val_set,
            callbacks=[checkpoint_callback],
        )

    def test_can_load_model(self):
        model = self.load_model()
        warnings.warn("Model loaded")

    def test_can_predict(self):
        model = self.load_model()
        preds = CatPredictor.predict_cat(model, self.val_set)
        self.assertIsNotNone(preds)
