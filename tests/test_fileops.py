import unittest
from click.testing import CliRunner

from os import path
import cv2


from puzzlesolver import puzzlesolver
from puzzlesolver import cli
from puzzlesolver.imageprocess.fileops import read_image

class TestFileOps(unittest.TestCase):
    '''Tests for file operations'''
    def test_reads_image(self):
        img = read_image('/home/zeynep/Projects/dissertation/puzzle/assets/Castle.png')
        self.assertIsNotNone(img)
