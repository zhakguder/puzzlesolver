import argparse
import os
import sys
import warnings
from configparser import ConfigParser

from puzzlesolver.classifiers.callbacks import (checkpoint_callback,
                                                checkpoint_config)
from puzzlesolver.classifiers.cat import CatPredictor
from puzzlesolver.classifiers.util import TFRecordExtractor
from puzzlesolver.utils import get_project_root

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    import tensorflow as tf
    from tensorflow import keras


config = ConfigParser()
PROJECT_ROOT = get_project_root()
CONFIG_FILE = os.path.join(PROJECT_ROOT, "puzzlesolver/classifiers/config.ini")
config.read(CONFIG_FILE)

train_config = config["training"]
data_config = config["dataset"]
EPOCHS = int(train_config["epochs"])
DATA_FOLDER = os.path.join(PROJECT_ROOT, data_config["data_folder"])
TFRECORD = os.path.join(DATA_FOLDER, data_config["tfrecord"])


class main:
    @staticmethod
    def train(epochs=EPOCHS):

        cat_model = CatPredictor()
        cat_model.compile(
            optimizer="rmsprop",
            # loss=tf.losses.categorical_crossentropy,
            loss=tf.nn.softmax_cross_entropy_with_logits,
            metrics=["accuracy"],
        )
        # get data for training
        t = TFRecordExtractor(TFRECORD)
        train_set, val_set = t.extract_image()

        cat_model.fit(
            train_set,
            epochs=epochs,
            validation_data=val_set,
            callbacks=[checkpoint_callback],
        )

    @staticmethod
    def embed(img_path):
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    # train
    subparser = subparsers.add_parser("train", help="train cat classifier network")
    subparser.add_argument("--epochs", help="Number of epochs for training", type=int)
    subparser.set_defaults(func=main.train)

    # get embeddings
    subparser = subparsers.add_parser("embed", help="get embeddings of images")
    subparser.add_argument("--filename", help="absolute path of image to embed")
    subparser.set_defaults(func=main.embed)

    if len(sys.argv) <= 2:
        parser.print_help()
        sys.exit()

    (args, unknown) = parser.parse_known_args()
    func = args.func
    del args.func

    args = dict(filter(lambda x: x[1], vars(args).items()))
    func(**args)
