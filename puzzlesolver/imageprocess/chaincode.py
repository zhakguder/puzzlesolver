'''Module implements Freeman's chaincode'''
import cv2

from math import floor

from puzzlesolver.imageprocess.fileops import read_image


def _contour(img_path, **kwargs):
    """
    Finds and imposes external contours of an image.

    >>> find_contours("../assets/Castle.png", threshold=250)
    True

    Args:
        img_path (str): Path of source image
        Kwargs:
            threshold (int): Value passed to cv2.threshold. Pixels under threshold will be black
    Returns:
        chaincode (list of int): Freeman chaincode of input image
    """
    threshold = kwargs['threshold']
    image = read_image(img_path)
    img_shape = image.shape[-1]
    assert img_shape == 3, f'Input image should have 2 or 3 channels but found {img_shape}'

    if img_shape == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.threshold(blurred, threshold, 255, cv2.THRESH_BINARY)[1]
    contours = cv2.findContours(thresh.copy(),
                                cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    return contours

def _contour_to_chaincode(contour):
    '''Takes a contour array where each row represents (x,y) values and outputs Freeman chaincode'''
    contour = contour.reshape(-1, 2)
    chain_code = []
    len_chain = len(contour)
    for i in range(len_chain-1):
        f_diff_x, f_diff_y = contour[i+1]-contour[i]
        code = 1
        if f_diff_x > 0:
            if f_diff_y>0:
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
            if f_diff_y>0:
                code = 4
            elif f_diff_y == 0:
                code = 5
            else:
                code = 6
        chain_code.append(code)
    return chain_code

def contours_to_chaincodes(contours):
    chain_codes = []
    for contour in contours:
        chain_codes.append(_contour_to_chaincode(contour))
    return chain_codes
