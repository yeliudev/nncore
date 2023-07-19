# Copyright (c) Ye Liu. Licensed under the MIT License.

import os

import pytest

import nncore


def test_construct():
    cfg = nncore.Config()
    assert cfg.filename is None
    assert cfg.text == ''
    assert len(cfg) == 0
    with pytest.raises(TypeError):
        nncore.Config([0, 1])


def test_build_config():
    for filename in ('a.py', 'a.b.py', 'b.json', 'c.yaml'):
        cfg_file = os.path.join(os.path.dirname(__file__), 'data', filename)
        cfg = nncore.Config.from_file(cfg_file)
        assert isinstance(cfg, nncore.Config)
        assert cfg.filename == cfg_file

    with pytest.raises(FileNotFoundError):
        nncore.Config.from_file('no_such_file.py')
    with pytest.raises(IOError):
        nncore.Config.from_file(
            os.path.join(os.path.dirname(__file__), 'data', 'color.jpg'))


def test_dict():
    cfg_dict = dict(item1=[1, 2], item2=dict(a=0), item3=True, item4='test')

    for filename in ('a.py', 'b.json', 'c.yaml'):
        cfg_file = os.path.join(os.path.dirname(__file__), 'data', filename)
        cfg = nncore.Config.from_file(cfg_file)

        assert len(cfg) == 4
        assert set(cfg.keys()) == set(cfg_dict.keys())
        assert set(cfg.keys()) == set(cfg_dict.keys())
        for value in cfg.values():
            assert value in cfg_dict.values()
        for name, value in cfg.items():
            assert name in cfg_dict
            assert value in cfg_dict.values()
        assert cfg.item1 == cfg_dict['item1']
        assert cfg.item2 == cfg_dict['item2']
        assert cfg.item2.a == 0
        assert cfg.item3 == cfg_dict['item3']
        assert cfg.item4 == cfg_dict['item4']
        with pytest.raises(AttributeError):
            cfg.not_exist
        for name in ('item1', 'item2', 'item3', 'item4'):
            assert name in cfg
            assert cfg[name] == cfg_dict[name]
            assert cfg.get(name) == cfg_dict[name]
            assert cfg.get('not_exist') is None
            assert cfg.get('not_exist', 0) == 0
            with pytest.raises(KeyError):
                cfg['not_exist']
        assert 'item1' in cfg
        assert 'not_exist' not in cfg
        cfg.update(dict(item1=0))
        assert cfg.item1 == 0
        cfg.update(dict(item2=dict(a=1)))
        assert cfg.item2.a == 1


def test_setattr():
    cfg = nncore.Config()
    cfg.item1 = [1, 2]
    cfg.item2 = {'a': 0}
    cfg['item5'] = {'a': {'b': None}}
    assert cfg['item1'] == [1, 2]
    assert cfg.item1 == [1, 2]
    assert cfg['item2'] == {'a': 0}
    assert cfg.item2.a == 0
    assert cfg['item5'] == {'a': {'b': None}}
    assert cfg.item5.a.b is None
