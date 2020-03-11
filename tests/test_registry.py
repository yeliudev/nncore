# Copyright (c) Ye Liu. All rights reserved.

import pytest

import nncore


def test_registry():
    reg_name = 'cat'
    CATS = nncore.Registry(reg_name)
    assert CATS.name == reg_name
    assert CATS.modules == dict()
    assert len(CATS) == 0

    @CATS.register_module
    class BritishShorthair:
        pass

    assert len(CATS) == 1
    assert CATS.get('BritishShorthair') is BritishShorthair

    class Munchkin:
        pass

    CATS.register_module(Munchkin)
    assert len(CATS) == 2
    assert CATS.get('Munchkin') is Munchkin

    with pytest.raises(KeyError):
        CATS.register_module(Munchkin)

    with pytest.raises(KeyError):

        @CATS.register_module
        class BritishShorthair:
            pass

    assert CATS.get('PersianCat') is None
    assert repr(CATS) in [
        "Registry(name=cat, modules=['BritishShorthair', 'Munchkin'])",
        "Registry(name=cat, modules=['Munchkin', 'BritishShorthair'])"
    ]

    with pytest.raises(TypeError):
        CATS.register_module(0)


def test_build_object():
    BACKBONES = nncore.Registry('backbone')

    @BACKBONES.register_module
    class ResNet:

        def __init__(self, depth, stages=4):
            self.depth = depth
            self.stages = stages

    @BACKBONES.register_module
    class ResNeXt:

        def __init__(self, depth, stages=4):
            self.depth = depth
            self.stages = stages

    cfg = dict(type='ResNet', depth=50)
    model = nncore.build_object(cfg, BACKBONES)
    assert isinstance(model, ResNet)
    assert model.depth == 50 and model.stages == 4

    cfg = dict(type='ResNet', depth=50)
    model = nncore.build_object(cfg, BACKBONES, default_args={'stages': 3})
    assert isinstance(model, ResNet)
    assert model.depth == 50 and model.stages == 3

    cfg = dict(type='ResNeXt', depth=50, stages=3)
    model = nncore.build_object(cfg, BACKBONES)
    assert isinstance(model, ResNeXt)
    assert model.depth == 50 and model.stages == 3

    with pytest.raises(KeyError):
        cfg = dict(type='VGG')
        model = nncore.build_object(cfg, BACKBONES)
