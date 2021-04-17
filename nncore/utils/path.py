# Copyright (c) Ye Liu. All rights reserved.

import os
import os.path as osp
from platform import system
from shutil import copy2, copytree, move, rmtree


def expand_user(*args):
    """
    Expand users in paths.

    Args:
        *args (str): The paths to be expanded.

    Returns:
        list[str] or str: The expanded path or list of paths.
    """
    if len(args) == 1:
        return osp.expanduser(args[0])
    else:
        return [osp.expanduser(p) for p in args]


def abs_path(path):
    """
    Parse the absolute path from a relative path.

    Args:
        path (str): Path to the file or directory.

    Returns:
        str: The absolute path.
    """
    return osp.abspath(expand_user(path))


def dir_name(path):
    """
    Parse the directory name from a path.

    Args:
        path (str): Path to the file or directory.

    Returns:
        str: The parsed directory name.
    """
    return osp.dirname(expand_user(path))


def base_name(path):
    """
    Parse the base name from a path.

    Args:
        path (str): Path to the file or directory.

    Returns:
        str: The parsed base name.
    """
    return osp.basename(path)


def join(*args):
    """
    Combine strings into a path.

    Args:
        *args (str): The strings to be combined.

    Returns:
        str: The combined path.
    """
    return osp.join(*args)


def split_ext(path):
    """
    Split name and extension of a path.

    Args:
        path (str): Path to the file or directory.

    Returns:
        tuple[str]: The splitted name and extension.
    """
    return osp.splitext(base_name(path))


def pure_name(path):
    """
    Parse the filename without extension from a path.

    Args:
        path (str): Path to the file

    Returns:
        str: The parsed filename.
    """
    return split_ext(path)[0]


def pure_ext(path):
    """
    Parse the file extension from a path.

    Args:
        path (str): Path to the file.

    Returns:
        str: The parsed extension.
    """
    return split_ext(path)[1][1:]


def rename(old_path, new_path):
    """
    Rename a file or a directory.

    Args:
        old_path (str): Old path to the file or directory.
        new_path (str): New path to the file or directory.
    """
    os.rename(old_path, new_path)


def cp(src, dst, symlink=True):
    """
    Copy files on the disk.

    Args:
        src (str): Path to the source file or directory.
        dst (str): Path to the destination file or directory.
        symlink (bool, optional): Whether to create a new symlink instead of
            copying the file it points to. Default: ``True``.
    """
    src, dst = expand_user(src, dst)
    if is_dir(src):
        if is_dir(dst):
            dst = join(dst, base_name(src))
        copytree(src, dst, symlinks=symlink)
    else:
        copy2(src, dst, follow_symlinks=symlink)


def mv(src, dst):
    """
    Move files on the disk.

    Args:
        src (str): Path to the source file or directory.
        dst (str): Path to the destination file or directory.
    """
    src, dst = expand_user(src, dst)
    move(src, dst)


def is_dir(path, raise_error=False):
    """
    Check whether a directory exists.

    Args:
        path (str): Path to the directory.
        raise_error (bool, optional): Whether to raise an error if the
            directory is not found. Default: ``False``.

    Returns:
        bool: Whether the directory exists.
    """
    is_dir = osp.isdir(expand_user(path))
    if not is_dir and raise_error:
        raise NotADirectoryError("directory '{}' not found".format(path))
    return is_dir


def is_file(path, raise_error=False):
    """
    Check whether a file exists.

    Args:
        path (str): Path to the file.
        raise_error (bool, optional): Whether to raise an error if the file is
            not found. Default: ``False``.

    Returns:
        bool: Whether the file exists.
    """
    is_file = osp.isfile(expand_user(path))
    if not is_file and raise_error:
        raise FileNotFoundError("file '{}' not found".format(path))
    return is_file


def mkdir(dir_name, raise_error=False, keep_empty=False):
    """
    Create a leaf directory and all intermediate ones.

    Args:
        dir_name (str): Path to the directory.
        raise_error (bool, optional): Whether to raise an error if the
            directory exists. Default: ``False``.
        keep_empty (bool, optional): Whether to keep the directory empty.
            Default: ``False``.
    """
    assert isinstance(dir_name, str) and dir_name != ''
    dir_name = expand_user(dir_name)
    os.makedirs(dir_name, exist_ok=not raise_error)
    if keep_empty:
        for f in os.listdir(dir_name):
            os.remove(join(dir_name, f))


def remove(path, raise_error=False):
    """
    Remove a file or a directory.

    Args:
        path (str): Path to the file or directory.
        raise_error (bool, optional): Whether to raise an error if the file is
            not found. Default: ``False``.
    """
    if is_file(path):
        os.remove(path)
    elif is_dir(path):
        rmtree(path)
    elif raise_error:
        raise FileNotFoundError(
            "file or directory '{}' not found".format(path))


def symlink(src, dst, overwrite=True, raise_error=False):
    """
    Create a symlink from source to destination.

    Args:
        src (str): Source of the symlink.
        dst (str): Destination of the symlink.
        overwrite (bool, optional): Whether to overwrite the old symlink if
            exists. Default: ``True``.
        raise_error (bool, optional): Whether to raise an error if the platform
            does not support symlink. Default: ``False``.
    """
    if system() == 'Windows' and not raise_error:
        return

    if osp.lexists(dst):
        if not overwrite:
            raise FileExistsError("file '{}' exists".format(dst))
        os.remove(dst)

    os.symlink(src, dst)
