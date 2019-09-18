import cv2


def _plot_contour(image, contour, main_contour=False):
    """Displays a single contour on an image"""
    if main_contour:
        color = (255, 0, 0)
    else:
        color = (0, 0, 255)
    image = cv2.drawContours(image, [contour], 0, color, 3)
    return image


def plot_contours(image, similar_contours, main_contour, contour_list):
    """Args:
    :param:similar contours: list of indices of similar contours
    :param:main_counter: int indicating index of query contour in contour_list
    """

    image = _plot_contour(image, contour_list[main_contour], main_contour=True)
    for contour in similar_contours:
        contour = int(contour)
        image = _plot_contour(image, contour_list[contour])
    return image
