import os
import warnings
from configparser import ConfigParser

from puzzlesolver.utils import get_project_root

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    import tensorflow as tf
    from tensorflow import keras

PROJECT_ROOT = get_project_root()
CONFIG_FILE = os.path.join(PROJECT_ROOT, "puzzlesolver/classifiers/config.ini")

config = ConfigParser()
config.read(CONFIG_FILE)

checkpoint_config = config["checkpoint"]
WEIGHT_PATH = checkpoint_config["weight_path"]
WEIGHT_FILE = os.path.join(PROJECT_ROOT, WEIGHT_PATH)
SAVE_FREQ = int(checkpoint_config["save_freq"])

checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=WEIGHT_FILE,
    monitor="accuracy",
    save_best_only=True,
    save_weights_only=True,
)
