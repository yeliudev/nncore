# Copyright (c) Ye Liu. All rights reserved.

import os
import os.path as osp
from pathlib import Path
from platform import system
from shutil import copy2, copytree, move, rmtree

from .misc import recursive


@recursive()
def expand_user(path):
    """
    Expand user in a path.

    Args:
        path (str): The path to be expanded.

    Returns:
        str: The expanded path.
    """
    return osp.expanduser(path)


@recursive()
def abs_path(path):
    """
    Parse absolute path from a relative path.

    Args:
        path (str): Path to the file or directory.

    Returns:
        str: The parsed absolute path.
    """
    return osp.abspath(expand_user(path))


@recursive()
def dir_name(path):
    """
    Parse directory name from a path.

    Args:
        path (str): Path to the file or directory.

    Returns:
        str: The parsed directory name.
    """
    return osp.dirname(expand_user(path))


@recursive()
def base_name(path):
    """
    Parse base name from a path.

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


@recursive()
def pure_name(path):
    """
    Parse pure filename from a path.

    Args:
        path (str): Path to the file

    Returns:
        str: The parsed pure filename.
    """
    return split_ext(path)[0]


@recursive()
def pure_ext(path):
    """
    Parse file extension from a path.

    Args:
        path (str): Path to the file.

    Returns:
        str: The parsed file extension.
    """
    return split_ext(path)[1][1:]


@recursive()
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


@recursive()
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


def ls(path=None, ext=None, skip_hidden_files=True, join_path=False):
    """
    List all files in a directory.

    Args:
        path (str | None, optional): Path to the directory. If not specified,
            the current working path ``'.'`` will be used. Default: ``None``.
        ext (list[str] | str | None, optional): The file extension or list of
            file extensions to keep. If specified, all the other files will be
            discarded. Default: ``None``.
        skip_hidden_files (bool, optional): Whether to discard hidden files
            whose filenames start with '.'. Default: ``True``.
        join_path (bool, optional): Whether to return the joined path of files.
            Default: ``False``.

    Returns:
        list: The list of files.
    """
    if path is not None and is_file(path):
        return [path]

    files = os.listdir(path)

    if isinstance(ext, (list, tuple)):
        files = [f for f in files if any(f.endswith(e) for e in ext)]
    elif isinstance(ext, str):
        files = [f for f in files if f.endswith(ext)]
    elif ext is not None:
        raise TypeError("ext must be a list or str, but got '{}'".format(
            type(ext)))

    if skip_hidden_files:
        files = [f for f in files if not f.startswith('.')]

    if join_path:
        files = [join(path, f) for f in files]

    return files


def find(path, pattern, sort=True):
    """
    Recursively search for files in a directory.

    Args:
        path (str): Path to the directory.
        pattern (str): The pattern of file names.
        sort (bool, optional): Whether to sort the results. Default: ``True``.

    Returns:
        list: The list of found files.
    """
    out = [str(m) for m in Path(path).rglob(pattern)]
    if sort:
        out = sorted(out)
    return out


def rename(old_path, new_path):
    """
    Rename a file or directory.

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
    src, dst = expand_user((src, dst))
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
    src, dst = expand_user((src, dst))
    move(src, dst)


@recursive()
def mkdir(dir_name, raise_error=False, keep_empty=False, modify_path=False):
    """
    Create a leaf directory and all intermediate ones.

    Args:
        dir_name (str): Path to the directory.
        raise_error (bool, optional): Whether to raise an error if the
            directory exists. Default: ``False``.
        keep_empty (bool, optional): Whether to keep the directory empty.
            Default: ``False``.
        modify_path (bool, optional): Whether to add ``'_i'`` (where i is an
            accumulating integer starting from ``0``) to the end of the path if
            the directory exists. Default: ``False``.

    Returns:
        str: Path to the actually created directory.
    """
    assert isinstance(dir_name, str) and dir_name != ''
    dir_name = expand_user(dir_name)

    if is_dir(dir_name) and modify_path:
        tmp, i = dir_name, 0
        while is_dir(tmp):
            tmp = '{}_{}'.format(dir_name, i)
            i += 1
        dir_name = tmp

    os.makedirs(dir_name, exist_ok=not raise_error)

    if keep_empty:
        for f in ls(dir_name, join_path=True):
            remove(f)

    return dir_name


def same_dir(old_path, new_path):
    """
    Parse another file or directory in the same directory.

    Args:
        old_path (str): Old path to the file or directory.
        new_path (str): New relative path to the file or directory.
    """
    return join(dir_name(old_path), new_path)


@recursive()
def remove(path, raise_error=False):
    """
    Remove a file or directory.

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
