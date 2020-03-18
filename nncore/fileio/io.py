# Copyright (c) Ye Liu. All rights reserved.

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
        raise TypeError('unsupported format: {}'.format(file_format))


def load(file_obj, file_format=None, **kwargs):
    """
    Load data from json/yaml/pickle files.

    Args:
        file_obj (str or file-like object): name of the file or a file-like
            object
        file_format (str, optional): if not specified, the file format will be
            inferred from the file extension, otherwise use the specified one.
            Currently supported formats include `json`, `yaml/yml` and
            `pickle/pkl`.

    Returns:
        obj (any): the content from the file
    """
    file_format = file_format or file_obj.split('.')[-1]
    _check_format(file_format, file_handlers)

    handler = file_handlers[file_format]

    if isinstance(file_obj, str):
        return handler.load_from_path(file_obj, **kwargs)
    elif hasattr(file_obj, 'read'):
        return handler.load_from_fileobj(file_obj, **kwargs)
    else:
        raise TypeError(
            'file_obj must be a str of a file-like object, but got {}'.format(
                type(file_obj)))


def dump(obj, file_obj, file_format=None, **kwargs):
    """
    Dump data to json/yaml/pickle files.

    Args:
        obj (any): the python object to be dumped
        file_obj (str or file-like object): name of the dumped file or a
            file-like object
        file_format (str, optional): if not specified, the file format will be
            inferred from the file extension, otherwise use the specified one.
            Currently supported formats include `json`, `yaml/yml` and
            `pickle/pkl`.
    """
    file_format = file_format or file_obj.split('.')[-1]
    _check_format(file_format, file_handlers)

    handler = file_handlers[file_format]

    if isinstance(file_obj, str):
        handler.dump_to_path(obj, file_obj, **kwargs)
    elif hasattr(file_obj, 'write'):
        handler.dump_to_fileobj(obj, file_obj, **kwargs)
    else:
        raise TypeError(
            'file_obj must be a str of a file-like object, but got {}'.format(
                type(file_obj)))


def loads(string, file_format='pickle', **kwargs):
    """
    Load data from json/yaml/pickle string.

    Args:
        string (str or btyearray): string of the file
        file_format (str, optional): format of the string. Only supports
            `pickle/pkl` currently.

    Returns:
        obj (any): the content from the file
    """
    _check_format(file_format, ['pickle', 'pkl'])
    handler = file_handlers[file_format]
    return handler.load_from_str(string, **kwargs)


def dumps(obj, file_format='pickle', **kwargs):
    """
    Dump data to json/yaml/pickle string.

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
