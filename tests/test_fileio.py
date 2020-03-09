# Copyright (c) Ye Liu. All rights reserved.

import os
import os.path as osp
import tempfile

import pytest

import nncore


def _test_handler(file_format, test_obj, str_checker, mode='r+'):
    # dump to a string
    dump_str = nncore.dumps(test_obj, file_format=file_format)
    str_checker(dump_str)

    # load/dump with filenames
    tmp_filename = osp.join(tempfile.gettempdir(), 'nncore_test_dump')
    nncore.dump(test_obj, tmp_filename, file_format=file_format)
    assert osp.isfile(tmp_filename)
    load_obj = nncore.load(tmp_filename, file_format=file_format)
    assert load_obj == test_obj
    os.remove(tmp_filename)

    # automatically inference the file format from the given filename
    tmp_filename = osp.join(tempfile.gettempdir(),
                            'nncore_test_dump.' + file_format)
    nncore.dump(test_obj, tmp_filename)
    assert osp.isfile(tmp_filename)
    load_obj = nncore.load(tmp_filename)
    assert load_obj == test_obj
    os.remove(tmp_filename)


obj_for_test = [{'a': 'abc', 'b': 1}, 2, 'c']


def test_json():

    def json_checker(dump_str):
        assert dump_str in [
            '[{"a": "abc", "b": 1}, 2, "c"]', '[{"b": 1, "a": "abc"}, 2, "c"]'
        ]

    _test_handler('json', obj_for_test, json_checker)


def test_yaml():

    def yaml_checker(dump_str):
        assert dump_str in [
            '- {a: abc, b: 1}\n- 2\n- c\n', '- {b: 1, a: abc}\n- 2\n- c\n',
            '- a: abc\n  b: 1\n- 2\n- c\n', '- b: 1\n  a: abc\n- 2\n- c\n'
        ]

    _test_handler('yaml', obj_for_test, yaml_checker)


def test_pickle():

    def pickle_checker(dump_str):
        import pickle
        assert pickle.loads(dump_str) == obj_for_test

    _test_handler('pickle', obj_for_test, pickle_checker, mode='rb+')


def test_exception():
    test_obj = [{'a': 'abc', 'b': 1}, 2, 'c']

    with pytest.raises(TypeError):
        nncore.dump(test_obj, 'tmp.txt')
