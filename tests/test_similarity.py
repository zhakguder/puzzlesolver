'''Tests for embedding images'''

import unittest

from puzzlesolver import puzzlesolver
from puzzlesolver.embedding import text_representation as text_embedding
from puzzlesolver.similarity import jaccard
from datasketch import MinHash

class TestSimilarity(unittest.TestCase):
    '''Tests for embedding images'''
    def setUp(self):
        self.chaincodes = [[1, 2, 3, 3, 4, 0, 0, 2, 3, 3], [7, 8, 8, 8, 8, 0], [0,1,2,3,1,2,3,4,5], [1,1,1,7,8,8,8]]
        self.documents = text_embedding.chaincodes_to_documents(self.chaincodes)
        self.document_1 = self.documents[0]
        self.document_2 = self.documents[1]

    def test_can_get_signature_for_a_document(self):
        signature = jaccard.document_signature(self.document_2)
        self.assertIsInstance(signature, MinHash)

    def test_can_get_signatures_for_multiple_documents(self):
        signatures = jaccard.document_signatures(self.documents)
        self.assertIsInstance(signatures, list)
        self.assertIsInstance(signatures[0], MinHash)

    def test_can_get_similar_documents(self):
        signatures = jaccard.document_signatures(self.documents)
        lsh_indices = jaccard.create_buckets(signatures)
        similar_documents = jaccard.query(lsh_indices, signatures[1])
        print(f'similar_documents: {similar_documents}')
