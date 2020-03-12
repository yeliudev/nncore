# Copyright (c) Ye Liu. All rights reserved.

from .handlers import JsonHandler, PickleHandler, YamlHandler

file_handlers = {
    'json': JsonHandler(),
    'yaml': YamlHandler(),
    'yml': YamlHandler(),
    'pickle': PickleHandler(),
    'pkl': PickleHandler()
}


def load(filename, file_format=None):
    """
    Load data from json/yaml/pickle files.

    Args:
        filename (str): name of the file
        file_format (str, optional): if not specified, the file format will be
            inferred from the file extension, otherwise use the specified one.
            Currently supported formats include `json`, `yaml/yml` and
            `pickle/pkl`.

    Returns:
        obj (any): the content from the file
    """
    if not isinstance(filename, str):
        raise TypeError("'filename' must be a str")

    file_format = file_format or filename.split('.')[-1]
    if file_format not in file_handlers:
        raise TypeError('unsupported format: {}'.format(file_format))

    handler = file_handlers[file_format]
    return handler.load_from_path(filename)


def loads(bytes, file_format='pickle'):
    """
    Load data from json/yaml/pickle bytes objects.

    Args:
        bytes (str or btyearray): bytes object of the file
        file_format (str, optional): format of the bytes object. Only supports
            `pickle/pkl` currently.

    Returns:
        obj (any): the content from the file
    """
    if file_format not in ['pickle', 'pkl']:
        raise TypeError('unsupported format: {}'.format(file_format))

    handler = file_handlers[file_format]
    return handler.load_from_bytes(bytes)


def dump(obj, filename, file_format=None):
    """
    Dump data to json/yaml/pickle files.

    Args:
        obj (any): the python object to be dumped
        filename (str): name of the dumped file
        file_format (str, optional): if not specified, the file format will be
            inferred from the file extension, otherwise use the specified one.
            Currently supported formats include `json`, `yaml/yml` and
            `pickle/pkl`.
    """
    if not isinstance(filename, str):
        raise TypeError("'filename' must be a str")

    file_format = file_format or filename.split('.')[-1]
    if file_format not in file_handlers:
        raise TypeError('unsupported format: {}'.format(file_format))

    handler = file_handlers[file_format]
    handler.dump_to_path(obj, filename)


def dumps(obj, file_format='pickle'):
    """
    Dump data to json/yaml/pickle bytes objects.

    Args:
        obj (any): the python object to be dumped
        file_format (str, optional): format of the bytes object. Currently
            supported formats include `json`, `yaml/yml` and `pickle/pkl`.

    Returns:
        string (str): the string of the dumped file
    """
    if file_format not in file_handlers:
        raise TypeError('unsupported format: {}'.format(file_format))

    handler = file_handlers[file_format]
    return handler.dump_to_bytes(obj)
