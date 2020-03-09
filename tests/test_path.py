# Copyright (c) Ye Liu. All rights reserved.

import os.path as osp

import pytest

import nncore


def test_dir_exist():
    nncore.dir_exist(osp.dirname(__file__))
    assert not nncore.dir_exist('no_such_dir')
    with pytest.raises(NotADirectoryError):
        nncore.dir_exist('no_such_dir', raise_error=True)


def test_file_exist():
    assert nncore.file_exist(__file__)
    assert not nncore.file_exist('no_such_file.txt')
    with pytest.raises(FileNotFoundError):
        nncore.file_exist('no_such_file.txt', raise_error=True)
