# Copyright (c) Ye Liu. All rights reserved.

import pytest

import nncore


def test_is_seq_of():
    assert nncore.is_seq_of([1.0, 2.0, 3.0], float)
    assert nncore.is_seq_of([(1, ), (2, ), (3, )], tuple)
    assert nncore.is_seq_of((1.0, 2.0, 3.0), float)
    assert nncore.is_list_of([1.0, 2.0, 3.0], float)
    assert not nncore.is_seq_of((1.0, 2.0, 3.0), float, seq_type=list)
    assert not nncore.is_tuple_of([1.0, 2.0, 3.0], float)
    assert not nncore.is_seq_of([1.0, 2, 3], int)
    assert not nncore.is_seq_of((1.0, 2, 3), int)


def test_slice_list():
    in_list = [1, 2, 3, 4, 5, 6]
    assert nncore.slice_list(in_list, [1, 2, 3]) == [[1], [2, 3], [4, 5, 6]]
    assert nncore.slice_list(in_list, [len(in_list)]) == [in_list]
    with pytest.raises(TypeError):
        nncore.slice_list(in_list, 2.0)
    with pytest.raises(ValueError):
        nncore.slice_list(in_list, [1, 2])


def test_concat_list():
    assert nncore.concat_list([[1, 2]]) == [1, 2]
    assert nncore.concat_list([[1, 2], [3, 4, 5], [6]]) == [1, 2, 3, 4, 5, 6]
