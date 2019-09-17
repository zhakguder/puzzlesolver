import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    import tensorflow as tf
    from tensorflow import keras

FILTER_SIZE = (3, 3)
DENSE_SIZE = 64
NUM_CLASSES = 2


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
        layers.append(keras.layers.Dense(NUM_CLASSES, activation="softmax"))
        self.model_layers = layers

    def call(self, inputs):
        output = inputs
        for layer in self.model_layers:
            output = layer(output)
        return output


def predict(fname):
    return -1
