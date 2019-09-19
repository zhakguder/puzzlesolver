import cv2
import numpy as np


def read_image(image_path):
    return cv2.imread(image_path)


def cut_image(image_path):
    whole_image = read_image(image_path)
    n_rows = whole_image.shape[0]
    upper_zeros = int(np.ceil(n_rows / 2))
    upper = np.copy(whole_image)
    lower = np.copy(whole_image)
    upper = _set_zeros(upper, range(upper_zeros, n_rows))
    lower = _set_zeros(lower, range(0, upper_zeros))
    return upper, lower


def _set_zeros(image, row_range, col_range=None):
    """
    Set the given ranges to zero in the image. Sets those parts to white.

   :param row_range:
   :type row_range: range object
   :param col_range:
   :type col_range: range object
    """
    n_cols = image.shape[1]
    if col_range is None:
        col_range = range(0, n_cols)
    for col in col_range:
        image[row_range, col] = 0
    return image
