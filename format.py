from yapf.yapflib.yapf_api import FormatFile
from sys import argv

file_to_format = argv[1]

FormatFile(file_to_format, in_place = True)
