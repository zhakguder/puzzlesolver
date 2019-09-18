import os
import warnings
from configparser import ConfigParser
from pdb import set_trace

from puzzlesolver.classifiers.callbacks import checkpoint_callback
from puzzlesolver.classifiers.util import _TestImageEmbedPrepper
from puzzlesolver.utils import get_project_root

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    import tensorflow as tf
    from tensorflow import keras


FILTER_SIZE = (3, 3)
DENSE_SIZE = 64
NUM_CLASSES = 2


PROJECT_ROOT = get_project_root()
config = ConfigParser()
CONFIG_FILE = os.path.join(PROJECT_ROOT, "puzzlesolver/classifiers/config.ini")
config.read(CONFIG_FILE)


model_config = config["checkpoint"]
weight_path = model_config["weight_path"]
ABS_WEIGHT_PATH = os.path.join(PROJECT_ROOT, weight_path)


class CatPredictor(keras.models.Model):
    def __init__(self, filter_list=[32, 64, 64], dense_size=DENSE_SIZE):
        super(CatPredictor, self).__init__()
        filter_size = FILTER_SIZE
        layers = []
        for filter_size in filter_list[:-1]:
            layers.append(
                keras.layers.Conv2D(filter_size, FILTER_SIZE, activation="relu")
            )
            layers.append(keras.layers.MaxPooling2D((2, 2)))
        else:
            layers.append(
                keras.layers.Conv2D(filter_list[-1], FILTER_SIZE, activation="relu")
            )

        layers.append(keras.layers.Flatten())
        layers.append(keras.layers.Dense(DENSE_SIZE, activation="relu"))
        layers.append(keras.layers.Dense(NUM_CLASSES, activation="linear"))
        self.model_layers = layers

    def call(self, inputs):
        output = inputs
        for layer in self.model_layers:
            output = layer(output)
        return output

    @staticmethod
    def load_model(weight_path):
        model = CatPredictor()
        model.compile(
            optimizer="rmsprop",
            loss=tf.losses.categorical_crossentropy,
            metrics=["accuracy"],
        )
        model.load_weights(weight_path).expect_partial()
        return model

    @staticmethod
    def predict_cat(data, model_path=ABS_WEIGHT_PATH):
        model = CatPredictor.load_model(model_path)
        return model.predict(data)

    @staticmethod
    def embed_image(img_path, model_path=ABS_WEIGHT_PATH):
        image = _TestImageEmbedPrepper.prep_image(img_path)

        return CatPredictor.predict_cat(image)
