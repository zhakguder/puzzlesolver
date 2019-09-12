'''Module implements image fragment similarity'''
from os import path

from puzzlesolver.imageprocess import chaincode
from puzzlesolver.embedding import text_representation as text_embedding
from puzzlesolver.similarity import jaccard
from puzzlesolver.utils import halver

def image_compute_signatures(img_path):

    # compute chain codes for all fragments
    chaincodes = chaincode.chaincodes(img_path)
    documents = text_embedding.chaincodes_to_documents(chaincodes)
    signatures = jaccard.document_signatures(documents)
    return signatures, chaincodes


def image_query_similar_fragments(chaincodes, signatures, query_index, threshold):
    lsh_indices = jaccard.create_buckets(signatures, threshold)
    # find the chain code for query chain
    q_code = text_embedding.query_chaincode(chaincodes[query_index])
    q_document = text_embedding.chaincodes_to_documents([q_code])
    q_signature = jaccard.document_signatures(q_document)[0]

    similar_documents = jaccard.query(lsh_indices, q_signature, query_index, threshold)
    return similar_documents

def search_n_similar(n, img_path, query_index):
    my_halver = halver()
    signatures, chaincodes = image_compute_signatures(img_path)
    for threshold in my_halver:
        similar_documents = image_query_similar_fragments(chaincodes, signatures, query_index, threshold)
        if len(similar_documents) >= n:
            break
    return similar_documents
