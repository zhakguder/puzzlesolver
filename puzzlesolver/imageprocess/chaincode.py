"""Module implements Freeman's chaincode"""
from math import floor
from pdb import set_trace

import cv2

from puzzlesolver.imageprocess.fileops import read_image

THRESHOLD = 254
MIN_CONTOUR_LENGTH = 500
TOLERANCE = 100


def _clean_non_contours(contours, image_shape):
    width, height = image_shape
    max_contour_length = (width + height) * 2 - TOLERANCE
    contours = [
        x
        for x in contours
        if len(x) > MIN_CONTOUR_LENGTH and len(x) < max_contour_length
    ]
    return contours


def _contour(img_path, threshold=THRESHOLD):
    """
    Finds and imposes external contours of an image.

    >>> find_contours("../assets/Castle.png", threshold=250)
    True

    Args:
        img_path (str): Path of source image
        threshold (int): Value passed to cv2.threshold. Pixels under threshold will be black
    Returns:
        chaincode (list of int): Freeman chaincode of input image
    """
    _, gray = _read_image_gray(img_path, threshold)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.threshold(blurred, threshold, 255, cv2.THRESH_BINARY)[1]
    contours = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[0]
    contours = _clean_non_contours(contours, gray.shape)
    return contours


def _read_image_gray(img_path, threshold):

    image = read_image(img_path)
    img_shape = image.shape[-1]
    assert (
        img_shape == 3
    ), f"Input image should have 2 or 3 channels but found {img_shape}"

    if img_shape == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image, gray


def image_and_contour_list(img_path, threshold=THRESHOLD):
    image, gray = _read_image_gray(img_path, threshold)
    contours = _contour(img_path, threshold)
    return image, contours


def _contour_to_chaincode(contour):
    """Takes a contour array where each row represents (x,y) values and outputs Freeman chaincode"""
    contour = contour.reshape(-1, 2)
    chain_code = []
    len_chain = len(contour)
    for i in range(len_chain - 1):
        f_diff_x, f_diff_y = contour[i + 1] - contour[i]
        code = 1
        if f_diff_x > 0:
            if f_diff_y > 0:
                code = 2
            elif f_diff_y == 0:
                code = 1
            else:
                code = 8
        elif f_diff_x == 0:
            if f_diff_y > 0:
                code = 3
            elif f_diff_y < 0:
                code = 7
        else:
            if f_diff_y > 0:
                code = 4
            elif f_diff_y == 0:
                code = 5
            else:
                code = 6
        chain_code.append(code)
    return chain_code


def _contours_to_chaincodes(img_path):
    contours = _contour(img_path, threshold=THRESHOLD)
    chain_codes = []
    for contour in contours:
        chain_codes.append(_contour_to_chaincode(contour))
    return chain_codes


def chaincodes(img_path):
    """Calculates chain codes of the image by finding and using its contours"""
    return _contours_to_chaincodes(img_path)
