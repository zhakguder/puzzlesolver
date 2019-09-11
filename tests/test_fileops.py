import unittest
from click.testing import CliRunner

from os import path
import cv2


from puzzlesolver import puzzlesolver
from puzzlesolver import cli
from puzzlesolver.imageprocess.fileops import read_image

from puzzlesolver.utils import get_project_root

class TestFileOps(unittest.TestCase):
    '''Tests for file operations'''
    def setUp(self):
        self.project_root = get_project_root()
        self.img_path = path.join(self.project_root, 'assets', 'Castle.png')
    def test_reads_image(self):
        img = read_image(self.img_path)
        self.assertIsNotNone(img)
