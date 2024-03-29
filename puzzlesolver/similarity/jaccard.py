from datasketch import MinHash, MinHashLSH

from puzzlesolver.similarity import config

similarity_config = config["similarity"]
THRESHOLD = float(similarity_config["threshold"])


def document_signature(document):
    data = document.split()
    m = MinHash(num_perm=128)

    for datum in data:
        m.update(datum.encode("utf8"))
    return m


def document_signatures(documents):
    signatures = []
    for document in documents:
        signatures.append(document_signature(document))
    return signatures


def create_buckets(minhash_list, threshold=THRESHOLD):
    lsh = MinHashLSH(threshold=threshold, num_perm=128)
    for i, minhash in enumerate(minhash_list):
        lsh.insert(str(i), minhash)
    return lsh


def query(lsh, document_signature, query_index, threshold=THRESHOLD):
    result = lsh.query(document_signature)
    similar_indices = [ind for ind in result if ind != str(query_index)]

    return similar_indices
