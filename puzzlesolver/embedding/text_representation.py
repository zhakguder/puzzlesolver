from pdb import set_trace
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


def _chaincode_transition_count_dict(chaincode):
    '''
    >>> _chaincode_transition_to_count_dict([1,2,2,3])
    {'12': 1, '23': 2, '31': 1}
'''
    # consider the end and beginning of the object to account for its closedness
    count_dict = {}
    cnt = 0
    for i, current in enumerate(chaincode):
        if i == 0:
            prev = current
            cnt += 1
            continue

        if i == len(chaincode)-1:
            code = f'{current}{chaincode[0]}'
            count_dict[code] = cnt
        if current == prev:
            cnt += 1

        else:
            code = f'{prev}{current}'
            count_dict[code] = cnt
            prev = current
            cnt = 1

    return count_dict
