'''Tests for embedding images'''

import unittest

import cv2

from click.testing import CliRunner
from os import path
from pdb import set_trace

from puzzlesolver import puzzlesolver
from puzzlesolver import cli
from puzzlesolver.embedding import text_representation as text_embedding


class TestTextEmbed(unittest.TestCase):
    '''Tests for embedding images'''
    def setUp(self):
        self.chaincodes = [[1, 2, 3, 3, 4, 0, 0, 2, 3, 3], [7, 8, 8, 8, 8, 0]]

    def test_chaincodes_to_documents(self):
        '''Tets that chain codes are converted to documents'''
        documents = text_embedding.chaincodes_to_documents(self.chaincodes)
        self.assertEqual(len(self.chaincodes), len(documents))

    def test_chaincode_to_document(self):
        '''Test that a document is produced for a single chain code'''

        document = text_embedding._chaincode_to_document(self.chaincodes[0])
        self.assertIsInstance(document, str)
        self.assertIn('_', document)
        self.assertIn('31', document)

    def test_chaincode_transition_counts(self):
        '''Test that pretransition sequence length is calculated correctly'''
        counts = text_embedding._chaincode_transition_counts(
            self.chaincodes[0])
        self.assertListEqual(counts, [('12', '1'), ('23', '1'), ('34', '2'),
                                      ('40', '1'), ('02', '2'), ('23', '1'),
                                      ('31', '2')])
