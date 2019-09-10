import unittest
from click.testing import CliRunner

from os import path
import cv2


from puzzlesolver import puzzlesolver
from puzzlesolver import cli
from puzzlesolver.imageprocess import chaincode
from pdb import set_trace


class TestImageProcess(unittest.TestCase):
    def setUp(self):
        self.img_path = '/home/zeynep/Projects/dissertation/puzzle/assets/Castle.png'
        self.threshold = 254
    '''Tests for imageprocess'''
    def test_contour_returns_something(self):
        test_contour = chaincode._contour(self.img_path, threshold=self.threshold)
        self.assertIsNotNone(test_contour)

    def test_contour_to_chaincode_returns_chaincode(self):
        '''Test that a chaincode is returned for a single contour'''
        test_contour = chaincode._contour(self.img_path, threshold=self.threshold)
        test_chaincode = chaincode._contour_to_chaincode(test_contour[1][0])
        self.assertLess(max(test_chaincode), 8)
        self.assertGreaterEqual( min(test_chaincode), 0)

    @unittest.expectedFailure
    def test_contours_to_chaincodes_returns_chaincodes(self):
        '''Test that a chaincode is returned for every contour'''
        test_contours = chaincode.contour(self.image_path, threshold=self.threshold)
        test_chaincodes = chaincode.contours_to_chaincodes(test_contours)
        self.assertEqual(len(test_contours), len(test_chaincodes))
