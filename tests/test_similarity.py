'''Tests for embedding images'''

import unittest

from datasketch import MinHash

from puzzlesolver import puzzlesolver
from puzzlesolver.embedding import text_representation as text_embedding

class TestTextEmbed(unittest.TestCase):
    '''Tests for embedding images'''
    def setUp(self):
        self.chaincodes = [[1, 2, 3, 3, 4, 0, 0, 2, 3, 3], [7, 8, 8, 8, 8, 0], [0,1,2,3,1,2,3,4,5], [1,1,1,7,8,8,8]]
        self.documents = text_embedding.chaincodes_to_documents(self.chaincodes)

    def test_can_start_bins(self):
        m1, m2 = MinHash(), MinHash()
        data1 = self.documents[1].split()
        data2 = self.documents[3].split()
        for d in data1:
            m1.update(d.encode('utf8'))
        for d in data2:
            m2.update(d.encode('utf8'))
        print("Estimated Jaccard for data1 and data2 is", m1.jaccard(m2))
