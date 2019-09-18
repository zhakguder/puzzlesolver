"""Module implements image fragment similarity"""
from os import path

import cv2

from puzzlesolver.embedding import text_representation as text_embedding
from puzzlesolver.imageprocess import chaincode, contour_ops
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
        similar_documents = image_query_similar_fragments(
            chaincodes, signatures, query_index, threshold
        )
        if len(similar_documents) >= n:
            break
    return similar_documents, threshold


def search_n_similars_all_contours(img_path, n):
    image, contours = chaincode.image_and_contour_list(img_path)
    most_similars = []
    for query_index, _ in enumerate(contours):
        if query_index > 0:
            image, contours = chaincode.image_and_contour_list(img_path)

        candidate_fragments, threshold = search_n_similar(n, img_path, query_index)

        print(
            f"Approximate neighbours to {query_index} with Jaccard similarity > {threshold}",
            candidate_fragments,
        )
        most_similars.append(candidate_fragments)
        contour_ops.plot_contours(image, candidate_fragments, query_index, contours)
        cv2.imwrite(f"similar_to_{query_index}.png", image)
    return most_similars
