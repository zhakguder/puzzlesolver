import os
from configparser import ConfigParser

from puzzlesolver.classifiers.callbacks import (checkpoint_callback,
                                                checkpoint_config)
from puzzlesolver.classifiers.cat import CatPredictor
from puzzlesolver.utils import get_project_root

config = ConfigParser()
PROJECT_ROOT = get_project_root()
train_config = config["training"]
EPOCHS = int(train_config["epochs"])
DATA_FOLDER = os.path.join(PROJECT_ROOT, train_config["data_folder"])
TFRECORD = os.path.join(DATA_FOLDER, train_config["tfrecord"])


def main():
    cat_model = CatPredictor()
    cat_model.compile(
        optimizer="rmsprop",
        # loss=tf.losses.categorical_crossentropy,
        loss=tf.nn.softmax_cross_entropy_with_logits,
        metrics=["accuracy"],
    )
    # get data for training
    test_TFRecord_path = os.path.join(data_folder, TFRECORD_PATH)
    t = TFRecordExtractor(test_TFRecord_path)
    train_set, val_set = t.extract_image()

    cat_model.fit(
        train_set,
        epochs=EPOCHS,
        validation_data=val_set,
        callbacks=[checkpoint_callback],
    )


if __name__ == "__main__":
    main()
