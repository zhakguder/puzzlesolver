'''Module implements image fragment similarity'''
from os import path

from puzzlesolver.imageprocess import chaincode
from puzzlesolver.embedding import text_representation as text_embedding

from puzzlesolver.similarity import jaccard


def image_compute_signatures(img_path):

    # compute chain codes for all fragments
    chaincodes = chaincode.chaincodes(img_path)
    documents = text_embedding.chaincodes_to_documents(chaincodes)
    signatures = jaccard.document_signatures(documents)
    lsh_indices = jaccard.create_buckets(signatures)
    return signatures, chaincodes, lsh_indices


def image_query_similar_fragments(chaincodes, lsh_indices, query_index):
    # find the chain code for query chain
    q_code = text_embedding.query_chaincode(chaincodes[query_index])
    q_document = text_embedding.chaincodes_to_documents([q_code])
    q_signature = jaccard.document_signatures(q_document)[0]


    similar_documents = jaccard.query(lsh_indices, q_signature, query_index)
    print(f'similar_documents: {similar_documents}')
    return similar_documents
