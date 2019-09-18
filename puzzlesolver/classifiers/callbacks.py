import os
import warnings

from puzzlesolver.classifiers import PROJECT_ROOT, config
from puzzlesolver.utils import get_project_root

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    import tensorflow as tf
    from tensorflow import keras

checkpoint_config = config["checkpoint"]
WEIGHT_PATH = checkpoint_config["weight_path"]
WEIGHT_FILE = os.path.join(PROJECT_ROOT, WEIGHT_PATH)
SAVE_FREQ = int(checkpoint_config["save_freq"])

checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=WEIGHT_FILE,
    monitor="val_loss",
    save_best_only=True,
    save_weights_only=True,
)
