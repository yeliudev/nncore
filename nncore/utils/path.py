# Copyright (c) Ye Liu. All rights reserved.

import os
import os.path as osp
from platform import system
from shutil import rmtree


def abs_path(path):
    """
    Parse the absolute path from a relative path.

    Args:
        path (str): path to the file or directory

    Returns:
        abs_path (str): the absolute path
    """
    return osp.abspath(path)


def base_name(path):
    """
    Parse the base name from a path.

    Args:
        path (str): path to the file or directory

    Returns:
        base_name (str): the parsed base name
    """
    return osp.basename(path)


def dir_name(path):
    """
    Parse the directory name from a path.

    Args:
        path (str): path to the file or directory

    Returns:
        dir_name (str): the parsed directory name
    """
    return osp.dirname(path)


def join(*args):
    """
    Combine strings into a path.

    Args:
        *args: strings to be combined

    Returns:
        path (str): the combined path
    """
    return osp.join(*args)


def split_ext(path):
    """
    Split name and extension of a path.

    Args:
        path (str): path to the file or directory

    Returns:
        name_ext (tuple[str]): the splitted name and extension
    """
    return osp.splitext(path)


def pure_name(path):
    """
    Parse the filename without extension from a path.

    Args:
        path (str): path to the file

    Returns:
        pure_name (str): the parsed filename
    """
    return osp.splitext(osp.basename(path))[0]


def pure_ext(path):
    """
    Parse the file extension from a path.

    Args:
        path (str): path to the file

    Returns:
        pure_ext (str): the parsed extension
    """
    return osp.splitext(osp.basename(path))[1]


def dir_exist(path, raise_error=False):
    """
    Check whether a directory exists.

    Args:
        path (str): (positive or relative) path to the directory
        raise_error (bool, optional): if `True`, raise an error when the file
            is not found

    Returns:
        is_dir (bool): whether the directory exists
    """
    is_dir = osp.isdir(path)
    if not is_dir and raise_error:
        raise NotADirectoryError("directory '{}' not found".format(path))
    return is_dir


def file_exist(path, raise_error=False):
    """
    Check whether a file exists.

    Args:
        path (str): (positive or relative) path to the file
        raise_error (bool, optional): if `True`, raise an error when the file
            is not found

    Returns:
        is_file (bool): whether the file exists
    """
    is_file = osp.isfile(path)
    if not is_file and raise_error:
        raise FileNotFoundError("file '{}' not found".format(path))
    return is_file


def mkdir(dir_name, exist_ok=True, keep_empty=False):
    """
    Create a leaf directory and all intermediate ones.

    Args:
        dir_name (str): (positive or relative) path to the directory
        exist_ok (bool, optional): if `False`, raise an error if the directory
            exists
        keep_empty (bool, optional): if `True`, remove all files in the
            directory
    """
    assert isinstance(dir_name, str) and dir_name != ''
    dir_name = osp.expanduser(dir_name)
    os.makedirs(dir_name, exist_ok=exist_ok)
    if keep_empty:
        for f in os.listdir(dir_name):
            os.remove(join(dir_name, f))


def remove(path, raise_error=False):
    """
    Remove a file or directory.

    Args:
        path (str): (positive or relative) path to the file or directory
        raise_error (bool, optional): if `True`, raise an error when the file
            is not found
    """
    if file_exist(path):
        os.remove(path)
    elif dir_exist(path):
        rmtree(path)
    elif raise_error:
        raise FileNotFoundError(
            "file or directory '{}' not found".format(path))


def symlink(src, dst, overwrite=True, raise_error=False):
    """
    Create a symlink from source to destination.

    Args:
        src (str): source of the symlink
        dst (str): destination of the symlink
        overwrite (bool, optional): if `True`, overwrite it if an old symlink
            exists
        raise_error (bool, optional): if `True`, raise an error if the platform
            does not support symlink
    """
    if system() == 'Windows' and not raise_error:
        return

    if osp.lexists(dst):
        if not overwrite:
            raise FileExistsError("file '{}' exists".format(dst))
        os.remove(dst)

    os.symlink(src, dst)
