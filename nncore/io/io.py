# Copyright (c) Ye Liu. All rights reserved.

import inspect
from functools import wraps

import h5py
import numpy as np

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

_open = open


def _get_handler(format):
    if format not in _FILE_HANDLERS:
        raise TypeError("unsupported format: '{}'".format(format))
    return _FILE_HANDLERS[format]


def load(name_or_file, format=None, **kwargs):
    """
    Load data from json/yaml/pickle/hdf5 files.

    Args:
        name_or_file (str or file object): Path to the file or a file object.
        format (str, optional): Format of the file. If not specified, the file
            format will be inferred from the file extension. Currently
            supported formats include ``json``, ``yaml/yml``, ``pickle/pkl``
            and ``hdf5/h5``. Default: ``None``.

    Returns:
        any: The loaded data.
    """
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


def dump(obj, name_or_file, format=None, overwrite=False, **kwargs):
    """
    Dump data to json/yaml/pickle/hdf5 files.

    Args:
        obj (any): The object to be dumped.
        name_or_file (str or file object): Path to the file or a file object.
        format (str, optional): Format of the file. If not specified, the file
            format will be inferred from the file extension. Currently
            supported formats include ``json``, ``yaml/yml``, ``pickle/pkl``
            and ``hdf5/h5``. Default: ``None``.
        overwrite (bool, optional): Whether to overwrite it if the file exists.
            Default: ``False``.
    """
    format = format or nncore.pure_ext(name_or_file)
    if format in ('hdf5', 'h5') and not isinstance(obj, np.ndarray):
        raise TypeError(
            "obj must be an np.ndarray for hdf5 files, but got '{}'".format(
                type(obj)))

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
    Load data from a json/yaml/pickle style string.

    Args:
        string (str or btyearray): String of the data.
        format (str, optional): Format of the string. Currently supported
            formats include ``json``, ``yaml/yml`` and ``pickle/pkl``. Default:
            ``'pickle'``.

    Returns:
        any: The loaded data.
    """
    assert format not in ('hdf5', 'h5')
    handler = _get_handler(format)
    return handler.load_from_str(string, **kwargs)


def dumps(obj, format='pickle', **kwargs):
    """
    Dump data to a json/yaml/pickle style string.

    Args:
        obj (any): The object to be dumped.
        format (str, optional): Format of the string. Currently supported
            formats include ``json``, ``yaml/yml`` and ``pickle/pkl``. Default:
            ``'pickle'``.

    Returns:
        str: The dumped string.
    """
    assert format not in ('hdf5', 'h5')
    handler = _get_handler(format)
    return handler.dump_to_str(obj, **kwargs)


def list_from_file(filename, offset=0, separator=',', max_length=-1):
    """
    Load a text file and parse the content as a list of tuples or str.

    Args:
        filename (str): Path to the file to be loaded.
        offset (int, optional): The offset of line numbers. Default: ``0``.
        separator (str or None, optional): The separator to use for parsing
            tuples. If not specified, each line would be treated as a str.
            Default: ``','``.
        max_length (int, optional): The maximum number of lines to be loaded.
            ``-1`` means all the lines from the file will be loaded. Default:
            ``-1``.

    Returns:
        list[str]: The loaded str list.
    """
    out_list, count = [], 0
    with _open(filename, 'r') as f:
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


def open(file=None, mode='r', format=None, as_decorator=None, **_kwargs):
    """
    Open a file and return a file object. This method can be used as a function
    or a decorator. When used as a decorator, the function to be decorated
    should receive the handler using an argument named ``f``. File and mode can
    be overrided during calls using the arguments ``file`` and ``mode``.
    Argument ``file`` and ``format`` should not be ``None`` at the same time.

    Args:
        file (str or None, optional): Path to the file to be loaded.
        mode (str, optional): The loading mode to use. Default: ``'r'``.
        format (str, optional): Format of the file. If not specified, the file
            format will be inferred from the file extension. Currently
            supported formats include ``json``, ``yaml/yml``, ``pickle/pkl``
            and ``hdf5/h5``. Default: ``None``.
        as_decorator (bool or None, optional): Whether this method is used as
            a decorator. Please explicitly assign a bool value to this
            argument when using this method in a Python Shell. If not
            specified, the method will try to determine it automatically.

    Returns:
        file-like object: The opened file object.
    """
    assert file is not None or format is not None
    format = format or nncore.pure_ext(file)

    if format in ('hdf5', 'h5'):
        handler = h5py.File
    else:
        handler = _open

    if as_decorator is None:
        context = inspect.stack()[1]
        as_decorator = context is not None and '@' in context[4][0]

    if not as_decorator:
        nncore.mkdir(nncore.dir_name(nncore.abs_path(file)))
        return handler(file, mode, **_kwargs)

    def _decorator(func):
        vars = func.__code__.co_varnames
        if 'file' in vars or 'mode' in vars:
            raise AttributeError(
                "decorated function should not have 'file' or 'mode' arguments"
            )

        @wraps(func)
        def _wrapper(*args, file=file, mode=mode, **kwargs):
            nncore.mkdir(nncore.dir_name(nncore.abs_path(file)))
            with handler(file, mode, **_kwargs) as f:
                func(*args, **kwargs, f=f)

        return _wrapper

    return _decorator
