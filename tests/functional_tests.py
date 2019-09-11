import unittest

from os import path

from datasketch import MinHash

from puzzlesolver.imageprocess import chaincode
from puzzlesolver.embedding import text_representation as text_embedding
from puzzlesolver.utils import get_project_root

from pdb import set_trace

class TestFunctional(unittest.TestCase):
    '''Functional tests'''
    def setUp(self):
        self.project_root = get_project_root()
        self.img_path = path.join(self.project_root, 'assets', 'Castle.png')
        self.threshold = 254

    def test_chaincodes_returns_chaincodes(self):
        chaincodes = chaincode.chaincodes(self.img_path)
