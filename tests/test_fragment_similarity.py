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
        self.query_index = 12
        self.lsh_threshold = 0.1
        self.n = 4

    def test_image_compute_signatures(self):
        signatures, chaincodes = fs.image_compute_signatures(self.img_path)
        self.assertIsNotNone(signatures)

    def test_can_return_query_fragments(self):
        signatures, chaincodes = fs.image_compute_signatures(self.img_path)
        similar_fragments = fs.image_query_similar_fragments(chaincodes, signatures, self.query_index, self.lsh_threshold)
        self.assertIsNotNone(similar_fragments)

    def test_can_return_n_similar(self):
        candidate_fragments = fs.search_n_similar(self.n, self.img_path, self.query_index)
