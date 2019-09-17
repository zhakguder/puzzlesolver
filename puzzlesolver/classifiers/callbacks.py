import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    import tensorflow as tf
    from tensorflow import keras


class WeightSavingCallback(tf.keras.callbacks.Callback):
    def __init__(self, weight_path):
        super(WeightSavingCallback, self).__init__()
        self.weight_path = weight_path

    def on_train_batch_end(self, batch, logs=None):
        self.model.save_weights(self.weight_path)
