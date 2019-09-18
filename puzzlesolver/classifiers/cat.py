import logging
import os
import warnings

from puzzlesolver.classifiers import PROJECT_ROOT, config
from puzzlesolver.classifiers.callbacks import checkpoint_callback
from puzzlesolver.classifiers.util import _TestImageEmbedPrepper

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    import tensorflow as tf
    from tensorflow import keras

FILTER_SIZE = (3, 3)
DENSE_SIZE = 64
NUM_CLASSES = 2


model_config = config["checkpoint"]
weight_path = model_config["weight_path"]
ABS_WEIGHT_PATH = os.path.join(PROJECT_ROOT, weight_path)

# logging.basicConfig(level=logging.INFO, filename="/tmp/output_embed", filemode="w")


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
        layers.append(keras.layers.Dense(DENSE_SIZE, activation="linear"))
        layers.append(keras.layers.Dense(NUM_CLASSES, activation="relu"))
        self.model_layers = layers

    def call(self, inputs):
        output = inputs
        # TODO unhack this
        n_layer = len(self.model_layers)
        for i, layer in enumerate(self.model_layers):
            output = layer(output)
            if i == n_layer - 2:
                print(output.numpy())
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
