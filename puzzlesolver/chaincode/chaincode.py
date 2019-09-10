def find_contours(img_path, **kwargs):
    """
    Finds and imposes external contours of an image.

    >>> find_contours("/puzzle/assets/Castle.png", threshold=250)
    True

    Args:
        img_path (str): Path of source image
        Kwargs:
            threshold (int): Value passed to cv2.threshold. Pixels under threshold will be black
    Returns:
        bool: The return value. True for success, False otherwise.
    """
    threshold = kwargs['threshold']
    image = cv2.imread(img_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.threshold(blurred, threshold, 255, cv2.THRESH_BINARY)[1]
    contours, _ = cv2.findContours(thresh.copy(),
    cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)


    contour_areas = [] # TODO: is there a faster way to keep the contors with n-largest areas?
    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        contour_areas.append((i, area))

        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        cv2.drawContours(image,[box],0,(0,0,255),2)
        cv2.imwrite('rect_{}.png'.format(i), image)
    # Contours that cover the largest areas are whole pieces
    largest_contour_areas = nlargest(10, contour_areas, key=lambda x:x[1])
    #print(largest_contour_areas)

    for i, area in largest_contour_areas:
        print (i, area)
        cv2.drawContours(image, contours, i, (100, 155, 100), 3)
        cv2.imwrite('contor_{}.png'.format(i), image)
    return True
