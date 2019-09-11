'''Tests for embedding images'''

import unittest

from lsh import LSHCache

from puzzlesolver import puzzlesolver
from puzzlesolver.embedding import text_representation as text_embedding

class TestTextEmbed(unittest.TestCase):
    '''Tests for embedding images'''
    def setUp(self):
        self.chaincodes = [[1, 2, 3, 3, 4, 0, 0, 2, 3, 3], [7, 8, 8, 8, 8, 0], [0,1,2,3,1,2,3,4,5], [1,1,1,7,8,8,8]]
        self.documents = text_embedding.chaincodes_to_documents(self.chaincodes)
        self.cache = LSHCache()

    @unittest.expectedFailure
    def test_can_start_bins(self):
        dups = {}
        for i, doc in enumerate(self.documents):
            dups[i] = self.cache.insert(doc.split(), i)
        print(dups)
        self.assertEqual(2,3)
