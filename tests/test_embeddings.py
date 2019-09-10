'''Tests for embedding images'''

import unittest

import cv2

from click.testing import CliRunner
from os import path
from pdb import set_trace

from puzzlesolver import puzzlesolver
from puzzlesolver import cli
from puzzlesolver.embedding import text_representation as text_embedding


class TestImageProcess(unittest.TestCase):
    '''Tests for embedding images'''
    def setUp(self):
        self.chaincodes = [[1,2,3,3,4,0], [7,8,8,8,8,0]]

    @unittest.expectedFailure
    def test_chaincodes_to_documents(self):
        '''Tets that chain codes are converted to documents'''
        documents = text_embedding.chaincodes_to_documents
        self.assertEqual(len(self.chaincodes), len(documents))

    @unittest.expectedFailure
    def test_chaincode_to_document(self):
        '''Test that a document is produced for a single chain code'''
        document = text_embedding._chaincode_to_document()
        self.assertIsInstance(document, str)
        self.assertIn('_', document)

    def test_chaincode_transition_count_dict(self):
        '''Test that pretransition sequence length is calculated correctly'''
        count_dict = text_embedding._chaincode_transition_count_dict(self.chaincodes[0])
        self.assertDictEqual(count_dict, {'12': 1, '23': 1, '34': 2, '40': 1, '01': 1})
