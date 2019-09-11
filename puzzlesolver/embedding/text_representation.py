'''Module implements methods to embed contour chain codes as text'''


def chaincodes_to_documents(chaincodes):
    '''Converts chain codes to documents
    Args:
    chaincodes (list of list of int)

    >>> chaincodes_to_documents([[1,2,2,3], [1,3,4]])
    ['12_1 23_2 31_1', '13_1 34_1 41_1']
'''
    documents = []
    for chaincode in chaincodes:
        documents.append(_chaincode_to_document(chaincode))
    return documents


def _chaincode_to_document(chaincode):
    '''Takes a chain code of an image and converts to a document
    Args:
    chaincode (list of int): Freeman chaincode

    Returns:
    document (str): each word in the document is separated by "_", left hand of "_" is the transition in the chain code, right hand of the chain code is the number of occurences of pretransition number.
    E.g 27_4 : represents 22227 in chain code

    >>> _chaincode_to_document([1,2,2,3])
    '12_1 23_2 31_1'
'''

    counts = _chaincode_transition_counts(chaincode)
    document = ''

    for transition, count in counts:
        document += f'{transition}_{transition} '
    return document


def _chaincode_transition_counts(chaincode):
    '''
    Counts number of pretransition codes in the chain code
    >>> _chaincode_transition_to_counts([1,2,2,3])
    [('12', '1'), ('23': '2'), ('31': '1')]

    Return:
    counts (tuple): first entry is a transition and second is the number of occurences of pretransition code
'''
    counts = []
    cnt = 0
    for i, current in enumerate(chaincode):
        if i == 0:
            prev = current
            cnt += 1
            continue

        if current == prev:
            cnt += 1
            prev = current
        # consider the end and beginning of the object to account for its closedness
        if i == len(chaincode) - 1:
            code = f'{current}{chaincode[0]}'
            counts.append((code, str(cnt)))
        elif current != prev:
            code = f'{prev}{current}'
            counts.append((code, str(cnt)))
            prev = current
            cnt = 1

    return counts
