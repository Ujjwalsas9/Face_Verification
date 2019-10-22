from mimetypes import MimeTypes as Mime
from constants import mimetypeList
import os


def file_type_validation(file1, file2):
    """
    function to validate the file type as image type
    :param file1: first input file
    :param file2: second input file
    :return:
    """
    if (file1 is not None) and (file2 is not None):
        if os.path.isfile(file1) and os.path.isfile(file2):
            if Mime().guess_type(file1)[0] in mimetypeList and Mime().guess_type(file2)[0] in mimetypeList:
                return 'success'
            return 'Invalid image type'
        return 'Image file not found'
    return 'Image should not be "None"'
