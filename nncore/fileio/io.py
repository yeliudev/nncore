# Copyright (c) Ye Liu. All rights reserved.

import os.path as osp

import nncore
from .handlers import JsonHandler, PickleHandler, YamlHandler

file_handlers = {
    'json': JsonHandler(),
    'yaml': YamlHandler(),
    'yml': YamlHandler(),
    'pickle': PickleHandler(),
    'pkl': PickleHandler()
}


def _check_format(file_format, supported_formats):
    if file_format not in supported_formats:
        raise TypeError("unsupported format: '{}'".format(file_format))


def load(name_or_file, file_format=None, **kwargs):
    """
    Load data from json/yaml/pickle files.

    Args:
        name_or_file (str or file-like object): name of the file or a file-like
            object
        file_format (str, optional): if not specified, the file format will be
            inferred from the file extension, otherwise use the specified one.
            Currently supported formats include `json`, `yaml/yml` and
            `pickle/pkl`.

    Returns:
        obj (any): the name_or_file from the file
    """
    file_format = file_format or name_or_file.split('.')[-1]
    _check_format(file_format, file_handlers)

    handler = file_handlers[file_format]

    if isinstance(name_or_file, str):
        return handler.load_from_path(name_or_file, **kwargs)
    elif hasattr(name_or_file, 'read'):
        return handler.load_from_fileobj(name_or_file, **kwargs)
    else:
        raise TypeError(
            "name_or_file must be a str or a file-like object, but got '{}'".
            format(type(name_or_file)))


def dump(obj, name_or_file, file_format=None, **kwargs):
    """
    Dump data to json/yaml/pickle files.

    Args:
        obj (any): the python object to be dumped
        name_or_file (str or file-like object): name of the dumped file or a
            file-like object
        file_format (str, optional): if not specified, the file format will be
            inferred from the file extension, otherwise use the specified one.
            Currently supported formats include `json`, `yaml/yml` and
            `pickle/pkl`.
    """
    file_format = file_format or name_or_file.split('.')[-1]
    _check_format(file_format, file_handlers)

    handler = file_handlers[file_format]

    if isinstance(name_or_file, str):
        nncore.mkdir(osp.dirname(name_or_file))
        handler.dump_to_path(obj, name_or_file, **kwargs)
    elif hasattr(name_or_file, 'write'):
        handler.dump_to_fileobj(obj, name_or_file, **kwargs)
    else:
        raise TypeError(
            "name_or_file must be a str or a file-like object, but got '{}'".
            format(type(name_or_file)))


def loads(string, file_format='pickle', **kwargs):
    """
    Load data from a json/yaml/pickle style string.

    Args:
        string (str or btyearray): string of the file
        file_format (str, optional): format of the string. Only supports
            `pickle/pkl` currently.

    Returns:
        obj (any): the name_or_file from the file
    """
    _check_format(file_format, ['pickle', 'pkl'])
    handler = file_handlers[file_format]
    return handler.load_from_str(string, **kwargs)


def dumps(obj, file_format='pickle', **kwargs):
    """
    Dump data to a json/yaml/pickle style string.

    Args:
        obj (any): the python object to be dumped
        file_format (str, optional): format of the string. Currently supported
            formats include `json`, `yaml/yml` and `pickle/pkl`.

    Returns:
        string (str): the string of the dumped file
    """
    _check_format(file_format, file_handlers)
    handler = file_handlers[file_format]
    return handler.dump_to_str(obj, **kwargs)


def list_from_file(filename, offset=0, separator=',', max_num=-1):
    """
    Load a text file and parse the content as a list of tuples or strings.

    Args:
        filename (str): name of the file to be loaded
        offset (int, optional): the offset of lines
        separator (str or None, optional): the separator to be used to parse
            tuples. If None, each line would be treated as a string.
        max_num (int, optional): the maximum number of lines to be read

    Returns:
        out_list (list[str]): A list of strings.
    """
    out_list, count = [], 0
    with open(filename, 'r') as f:
        for _ in range(offset):
            f.readline()
        for line in f:
            if max_num >= 0 and count >= max_num:
                break
            line = line.rstrip('\n')
            if separator is not None:
                line = tuple(line.split(separator))
            out_list.append(line)
            count += 1
    return out_list
