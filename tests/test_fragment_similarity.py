import unittest

from os import path

from puzzlesolver.embedding import text_representation as text_embedding
from puzzlesolver.imageprocess import chaincode
from puzzlesolver.similarity import fragment_similarity as fs
from puzzlesolver.utils import get_project_root



class TestFragmentSimilarity(unittest.TestCase):
    '''Tests for finding similar image fragments'''
    def setUp(self):
        self.project_root = get_project_root()
        self.img_path = path.join(self.project_root, 'assets', 'Castle.png')
        self.threshold = 254

    def test_image_compute_signatures(self):
        signatures, chaincodes, lsh_indices = fs.image_compute_signatures(self.img_path)
        self.assertIsNotNone(signatures)

    def test_can_return_query_fragments(self):
        QUERY_INDEX = 12
        signatures, chaincodes, lsh_indices = fs.image_compute_signatures(self.img_path)
        similar_fragments = fs.image_query_similar_fragments(chaincodes, lsh_indices, QUERY_INDEX)
        self.assertIsNotNone(similar_fragments)
