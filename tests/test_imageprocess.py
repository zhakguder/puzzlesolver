'''Tests for image processing'''

import unittest

import cv2

from click.testing import CliRunner
from os import path

from puzzlesolver.imageprocess import chaincode
from puzzlesolver.utils import get_project_root


class TestImageProcess(unittest.TestCase):
    '''Tests for imageprocess'''
    def setUp(self):
        self.project_root = get_project_root()
        self.img_path = path.join(self.project_root, 'assets', 'Castle.png')
        self.threshold = 254


    def test_contours_to_chaincodes_returns_chaincodes(self):
        '''Test that a chaincode is returned for every contour'''
        test_contours = chaincode._contour(self.img_path,
                                          threshold=self.threshold)
        test_chaincodes = chaincode._contours_to_chaincodes(self.img_path)
        self.assertEqual(len(test_contours), len(test_chaincodes))

    def test_contour_to_chaincode_returns_chaincode(self):
        '''Test that a chaincode is returned for a single contour'''
        test_contour = chaincode._contour(self.img_path,
                                          threshold=self.threshold)
        test_chaincode = chaincode._contour_to_chaincode(test_contour[0])
        self.assertLess(max(test_chaincode), 9)
        self.assertGreater(min(test_chaincode), 0)

    def test_contour_returns_something(self):
        test_contour = chaincode._contour(self.img_path,
                                          threshold=self.threshold)
        self.assertIsNotNone(test_contour)
