# Copyright (c) Ye Liu. Licensed under the MIT License.

import os

import pytest

import nncore


def test_is_dir():
    nncore.is_dir(os.path.dirname(__file__))
    assert not nncore.is_dir('no_such_dir')
    with pytest.raises(NotADirectoryError):
        nncore.is_dir('no_such_dir', raise_error=True)


def test_is_file():
    assert nncore.is_file(__file__)
    assert not nncore.is_file('no_such_file.txt')
    with pytest.raises(FileNotFoundError):
        nncore.is_file('no_such_file.txt', raise_error=True)
