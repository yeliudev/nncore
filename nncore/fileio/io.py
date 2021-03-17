# Copyright (c) Ye Liu. All rights reserved.

import os.path as osp

import nncore
from .handlers import Hdf5Handler, JsonHandler, PickleHandler, YamlHandler

_FILE_HANDLERS = {
    'json': JsonHandler(),
    'yaml': YamlHandler(),
    'yml': YamlHandler(),
    'pickle': PickleHandler(),
    'pkl': PickleHandler(),
    'hdf5': Hdf5Handler(),
    'h5': Hdf5Handler()
}


def _get_handler(format):
    if format not in _FILE_HANDLERS:
        raise TypeError("unsupported format: '{}'".format(format))
    return _FILE_HANDLERS[format]


def load(name_or_file, format=None, **kwargs):
    """
    Load data from json/yaml/pickle/hdf5 files.

    Args:
        name_or_file (str or file-like object): name of the file or a file-like
            object
        format (str, optional): if not specified, the file format will be
            inferred from the file extension, otherwise use the specified one.
            Currently supported formats include `json`, `yaml/yml`,
            `pickle/pkl` and `hdf5/h5`.

    Returns:
        obj (any): the name_or_file from the file
    """
    format = format or name_or_file.split('.')[-1]
    handler = _get_handler(format)

    if isinstance(name_or_file, str):
        return handler.load_from_path(name_or_file, **kwargs)
    elif hasattr(name_or_file, 'read'):
        return handler.load_from_fileobj(name_or_file, **kwargs)
    else:
        raise TypeError(
            "name_or_file must be a str or a file-like object, but got '{}'".
            format(type(name_or_file)))


def dump(obj, name_or_file, format=None, **kwargs):
    """
    Dump data to json/yaml/pickle/hdf5 files.

    Args:
        obj (any): the python object to be dumped
        name_or_file (str or file-like object): name of the dumped file or a
            file-like object
        format (str, optional): if not specified, the file format will be
            inferred from the file extension, otherwise use the specified one.
            Currently supported formats include `json`, `yaml/yml`,
            `pickle/pkl` and `hdf5/h5`.
    """
    format = format or name_or_file.split('.')[-1]
    handler = _get_handler(format)

    if isinstance(name_or_file, str):
        nncore.mkdir(osp.dirname(name_or_file))
        handler.dump_to_path(obj, name_or_file, **kwargs)
    elif hasattr(name_or_file, 'write'):
        handler.dump_to_fileobj(obj, name_or_file, **kwargs)
    else:
        raise TypeError(
            "name_or_file must be a str or a file-like object, but got '{}'".
            format(type(name_or_file)))


def loads(string, format='pickle', **kwargs):
    """
    Load data from a json/yaml/pickle style string.

    Args:
        string (str or btyearray): string of the file
        format (str, optional): format of the string. Currently supported
            formats include `json`, `yaml/yml`, and `pickle/pkl`.

    Returns:
        obj (any): the name_or_file from the file
    """
    handler = _get_handler(format)
    return handler.load_from_str(string, **kwargs)


def dumps(obj, format='pickle', **kwargs):
    """
    Dump data to a json/yaml/pickle style string.

    Args:
        obj (any): the python object to be dumped
        format (str, optional): format of the string. Currently supported
            formats include `json`, `yaml/yml`, and `pickle/pkl`.

    Returns:
        string (str): the string of the dumped file
    """
    handler = _get_handler(format)
    return handler.dump_to_str(obj, **kwargs)


def list_from_file(filename, offset=0, separator=',', max_length=-1):
    """
    Load a text file and parse the content as a list of tuples or strings.

    Args:
        filename (str): name of the file to be loaded
        offset (int, optional): the offset of lines
        separator (str or None, optional): the separator to be used to parse
            tuples. If `None`, each line would be treated as a string.
        max_length (int, optional): the maximum number of lines to be loaded

    Returns:
        out_list (list[str]): A list of strings.
    """
    out_list, count = [], 0
    with open(filename, 'r') as f:
        for _ in range(offset):
            f.readline()
        for line in f:
            if max_length >= 0 and count >= max_length:
                break
            line = line.rstrip('\n')
            if separator is not None:
                line = tuple(line.split(separator))
            out_list.append(line)
            count += 1
    return out_list
