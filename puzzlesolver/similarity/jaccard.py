from datasketch import MinHash, MinHashLSH
THRESHOLD = 0.4

def document_signature(document):
    data = document.split()
    m = MinHash()

    for datum in data:
        m.update(datum.encode('utf8'))
    return m

def document_signatures(documents):
    signatures = []
    for document in documents:
        signatures.append(document_signature(document))
    return signatures

def create_buckets(minhash_list):
    lsh = MinHashLSH(threshold=THRESHOLD, num_perm=128)
    for i, minhash in enumerate(minhash_list):
        lsh.insert(str(i), minhash)
    return lsh

def query(lsh, document_signature):
    result = lsh.query(document_signature)
    print(f'Approximate neighbours with Jaccard similarity > {THRESHOLD}', result)
    return result
