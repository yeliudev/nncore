# Copyright (c) Ye Liu. All rights reserved.

import inspect
from functools import wraps

import h5py
import jsonlines

import nncore
from .handlers import (HDF5Handler, JSONHandler, JSONLHandler, NumPyHandler,
                       PickleHandler, TXTHandler, XMLHandler, YAMLHandler)

_FILE_HANDLERS = {
    'json': JSONHandler(),
    'jsonl': JSONLHandler(),
    'yaml': YAMLHandler(),
    'yml': YAMLHandler(),
    'pickle': PickleHandler(),
    'pkl': PickleHandler(),
    'hdf5': HDF5Handler(),
    'h5': HDF5Handler(),
    'npy': NumPyHandler(),
    'npz': NumPyHandler(),
    'xml': XMLHandler(),
    'txt': TXTHandler()
}

_open = open


def _get_handler(format):
    if format not in _FILE_HANDLERS:
        raise TypeError("unsupported format: '{}'".format(format))
    return _FILE_HANDLERS[format]


def load(name_or_file, format=None, **kwargs):
    """
    Load data from files.

    Args:
        name_or_file (list | str | file object): Paths to the files or file
            objects.
        format (str, optional): Format of the file. If not specified, the file
            format will be inferred from the file extension. Currently
            supported formats include ``json/jsonl``, ``yaml/yml``,
            ``pickle/pkl``, ``hdf5/h5``, ``npy/npz``, ``xml``, and ``txt``.
            Default: ``None``.

    Returns:
        any: The loaded data.
    """
    if isinstance(name_or_file, (list, tuple)):
        return [load(n, format=format, **kwargs) for n in name_or_file]

    format = format or nncore.pure_ext(name_or_file)
    handler = _get_handler(format)

    if isinstance(name_or_file, str):
        return handler.load_from_path(name_or_file, **kwargs)
    elif hasattr(name_or_file, 'close'):
        return handler.load_from_file(name_or_file, **kwargs)
    else:
        raise TypeError(
            "name_or_file must be a str or a file object, but got '{}'".format(
                type(name_or_file)))


def dump(obj, name_or_file, format=None, overwrite=True, **kwargs):
    """
    Dump data to a file.

    Args:
        obj (any): The object to be dumped.
        name_or_file (str | file object): Path to the file or a file object.
        format (str, optional): Format of the file. If not specified, the file
            format will be inferred from the file extension. Currently
            supported formats include ``json/jsonl``, ``yaml/yml``,
            ``pickle/pkl``, ``hdf5/h5``, ``npy/npz``, ``xml``, and ``txt``.
            Default: ``None``.
        overwrite (bool, optional): Whether to overwrite it if the file exists.
            Default: ``True``.
    """
    format = format or nncore.pure_ext(name_or_file)
    handler = _get_handler(format)

    if isinstance(name_or_file, str):
        if nncore.is_file(name_or_file):
            if overwrite:
                nncore.remove(name_or_file)
            else:
                raise FileExistsError("file '{}' exists".format(name_or_file))
        nncore.mkdir(nncore.dir_name(nncore.abs_path(name_or_file)))
        handler.dump_to_path(obj, name_or_file, **kwargs)
    elif hasattr(name_or_file, 'close'):
        handler.dump_to_file(obj, name_or_file, **kwargs)
    else:
        raise TypeError(
            "name_or_file must be a str or a file object, but got '{}'".format(
                type(name_or_file)))


def loads(string, format='pickle', **kwargs):
    """
    Load data from strings.

    Args:
        string (list | str | btyearray): Strings of the data.
        format (str, optional): Format of the string. Currently supported
            formats include ``json/jsonl``, ``yaml/yml``, ``pickle/pkl`` and
            ``xml``. Default: ``'pickle'``.

    Returns:
        any: The loaded data.
    """
    if isinstance(string, (list, tuple)):
        return [loads(s, format=format, **kwargs) for s in string]

    handler = _get_handler(format)
    return handler.load_from_str(string, **kwargs)


def dumps(obj, format='pickle', **kwargs):
    """
    Dump data to a string.

    Args:
        obj (any): The object to be dumped.
        format (str, optional): Format of the string. Currently supported
            formats include ``json/jsonl``, ``yaml/yml``, ``pickle/pkl`` and
            ``xml``. Default: ``'pickle'``.

    Returns:
        str: The dumped string.
    """
    handler = _get_handler(format)
    return handler.dump_to_str(obj, **kwargs)


def list_from_file(filename,
                   encoding='utf-8',
                   offset=0,
                   separator=',',
                   max_length=-1):
    """
    Load a text file and parse the content as a list of tuples or str.

    Args:
        filename (str): Path to the file to be loaded.
        encoding (str, optional): The encoding of the file.
        offset (int, optional): The offset of line numbers. Default: ``0``.
        separator (str | None, optional): The separator to use for parsing
            tuples. If not specified, each line would be treated as a str.
            Default: ``','``.
        max_length (int, optional): The maximum number of lines to be loaded.
            ``-1`` means all the lines from the file will be loaded. Default:
            ``-1``.

    Returns:
        list[str]: The loaded str list.
    """
    out_list, count = [], 0
    with _open(filename, 'r', encoding=encoding) as f:
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


def open(file=None, mode='r', format=None, as_decorator=None, **kwargs):
    """
    Open a file and return a file object. This method can be used as a function
    or a decorator. When used as a decorator, the function to be decorated
    should receive the handler using an argument named ``f``. File and mode can
    be overrided during calls using the arguments ``file`` and ``mode``.
    Argument ``file`` and ``format`` should not be ``None`` at the same time.

    Args:
        file (str | None, optional): Path to the file to be loaded.
        mode (str, optional): The loading mode to use. Default: ``'r'``.
        format (str, optional): Format of the file. If not specified, the file
            format will be inferred from the file extension. Currently
            supported formats include ``jsonl`` and ``hdf5/h5``. Default:
            ``None``.
        as_decorator (bool | None, optional): Whether this method is used as a
            decorator. Please explicitly assign a bool value to this argument
            when using this method in a Python Shell. If not specified, the
            method will try to determine it automatically.

    Returns:
        file object: The opened file object.
    """
    assert file is not None or format is not None
    format = format or nncore.pure_ext(file)

    if format in ('hdf5', 'h5'):
        handler = h5py.File
    elif format == 'jsonl':
        handler = jsonlines.open
    else:
        handler = _open

    if as_decorator is None:
        context = inspect.stack()[1]
        as_decorator = context is not None and '@' in context[4][0]

    if not as_decorator:
        nncore.mkdir(nncore.dir_name(nncore.abs_path(file)))
        return handler(file, mode, **kwargs)

    def _decorator(func):
        vars = func.__code__.co_varnames
        if 'file' in vars or 'mode' in vars:
            raise AttributeError(
                "decorated function should not have 'file' or 'mode' arguments"
            )

        @wraps(func)
        def _wrapper(*args, file=file, mode=mode, **_kwargs):
            nncore.mkdir(nncore.dir_name(nncore.abs_path(file)))
            with handler(file, mode, **kwargs) as f:
                func(*args, **_kwargs, f=f)

        return _wrapper

    return _decorator
