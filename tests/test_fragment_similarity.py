import unittest
from os import path

import cv2

from puzzlesolver.embedding import text_representation as text_embedding
from puzzlesolver.imageprocess import chaincode, contour_ops
from puzzlesolver.similarity import fragment_similarity as fs
from puzzlesolver.utils import get_project_root


class TestFragmentSimilarity(unittest.TestCase):
    """Tests for finding similar image fragments"""

    def setUp(self):
        self.project_root = get_project_root()
        self.img_path = path.join(self.project_root, "assets", "Castle.png")
        self.threshold = 254
        self.query_index = 1
        self.lsh_threshold = 0.1
        self.n = 1

    def test_image_compute_signatures(self):
        signatures, chaincodes = fs.image_compute_signatures(self.img_path)
        self.assertIsNotNone(signatures)

    def test_can_return_query_fragments(self):
        signatures, chaincodes = fs.image_compute_signatures(self.img_path)
        similar_fragments = fs.image_query_similar_fragments(
            chaincodes, signatures, self.query_index, self.lsh_threshold
        )
        self.assertIsNotNone(similar_fragments)

    def test_can_return_n_similar(self):
        candidate_fragments, threshold = fs.search_n_similar(
            self.n, self.img_path, self.query_index
        )

        print(
            f"Approximate neighbours to {self.query_index} with Jaccard similarity > {threshold}",
            candidate_fragments,
        )

    def test_can_plot_similar_contours(self):
        image, contours = chaincode.image_and_contour_list(self.img_path)
        for query_index, _ in enumerate(contours):
            if query_index > 0:
                image, contours = chaincode.image_and_contour_list(self.img_path)

            candidate_fragments, threshold = fs.search_n_similar(
                self.n, self.img_path, query_index
            )

            print(
                f"Approximate neighbours to {query_index} with Jaccard similarity > {threshold}",
                candidate_fragments,
            )
            contour_ops.plot_contours(image, candidate_fragments, query_index, contours)
            cv2.imwrite(f"similar_to_{query_index}.png", image)
