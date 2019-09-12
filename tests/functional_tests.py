import unittest

from os import path

from puzzlesolver.imageprocess import chaincode
from puzzlesolver.embedding import text_representation as text_embedding
from puzzlesolver.utils import get_project_root


from puzzlesolver.similarity import jaccard


class TestFunctional(unittest.TestCase):
    '''Functional tests'''
    def setUp(self):
        self.project_root = get_project_root()
        self.img_path = path.join(self.project_root, 'assets', 'Castle.png')
        self.threshold = 254

    def test_chaincodes_returns_chaincodes(self):
        chaincodes = chaincode.chaincodes(self.img_path)

    def test_minhash_chaincodes(self):
        QUERY_INDEX = 3

        # compute chain codes for all fragments
        chaincodes = chaincode.chaincodes(self.img_path)
        # find the chain code for query chain
        q_code = text_embedding.query_chaincode(chaincodes[QUERY_INDEX])

        documents = text_embedding.chaincodes_to_documents(chaincodes)
        q_document = text_embedding.chaincodes_to_documents([q_code])

        signatures = jaccard.document_signatures(documents)
        q_signature = jaccard.document_signatures(q_document)[0]

        lsh_indices = jaccard.create_buckets(signatures)
        similar_documents = jaccard.query(lsh_indices, q_signature, QUERY_INDEX)
        print(f'similar_documents: {similar_documents}')
        self.assertGreater(len(similar_documents), 0)
