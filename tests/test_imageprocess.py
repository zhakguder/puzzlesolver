"""Tests for image processing"""

import unittest
from os import path

import cv2
import numpy as np

from puzzlesolver.imageprocess import chaincode, fileops
from puzzlesolver.utils import get_project_root


class TestImageProcess(unittest.TestCase):
    """Tests for imageprocess"""

    def setUp(self):
        self.project_root = get_project_root()
        self.img_path = path.join(self.project_root, "assets", "Castle.png")
        self.threshold = 254
        self.dog_image = path.join(self.project_root, "data/train/dog.1526.jpg")

    def test_contours_to_chaincodes_returns_chaincodes(self):
        """Test that a chaincode is returned for every contour"""
        test_contours = chaincode._contour(self.img_path, threshold=self.threshold)
        test_chaincodes = chaincode._contours_to_chaincodes(self.img_path)
        self.assertEqual(len(test_contours), len(test_chaincodes))

    def test_contour_to_chaincode_returns_chaincode(self):
        """Test that a chaincode is returned for a single contour"""
        test_contour = chaincode._contour(self.img_path, threshold=self.threshold)
        test_chaincode = chaincode._contour_to_chaincode(test_contour[0])
        self.assertLess(max(test_chaincode), 9)
        self.assertGreater(min(test_chaincode), 0)

    def test_contour_returns_something(self):
        test_contour = chaincode._contour(self.img_path, threshold=self.threshold)
        self.assertIsNotNone(test_contour)

    @unittest.expectedFailure
    def test_can_cut_image(self):
        upper, lower = fileops.cut_image(self.dog_image)
        self.assertIsNotNone(upper)
        self.assertIsNotNone(lower)

    def test_can_set_mask(self):
        matrix = np.ones((10, 10))
        n_all_entries = np.sum(matrix)
        row_range = range(7, 10)
        col_range = range(3, 6)
        matrix = fileops._set_zeros(matrix, row_range, col_range)
        n_zeros = len(row_range) * len(col_range)
        self.assertEqual(np.sum(matrix), n_all_entries - n_zeros)

    def test_can_cut_image(self):
        upper, lower = fileops.cut_image(self.dog_image)
        cv2.imwrite("upper.png", upper)
        cv2.imwrite("lower.png", lower)
