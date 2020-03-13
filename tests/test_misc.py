# Copyright (c) Ye Liu. All rights reserved.

import pytest

import nncore


def test_iter_cast():
    assert nncore.list_cast([1, 2, 3], int) == [1, 2, 3]
    assert nncore.list_cast(['1.1', 2, '3'], float) == [1.1, 2.0, 3.0]
    assert nncore.list_cast([1, 2, 3], str) == ['1', '2', '3']
    assert nncore.tuple_cast((1, 2, 3), str) == ('1', '2', '3')
    assert next(nncore.iter_cast([1, 2, 3], str)) == '1'
    with pytest.raises(TypeError):
        nncore.iter_cast([1, 2, 3], '')
    with pytest.raises(TypeError):
        nncore.iter_cast(1, str)


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


def test_bind_getter():

    @nncore.bind_getter('name', 'depth')
    class Backbone:
        _name = 'ResNet'
        _depth = 50

    backbone = Backbone()
    assert backbone.name == 'ResNet'
    assert backbone.depth == 50
