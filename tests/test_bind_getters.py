# Copyright (c) Ye Liu. All rights reserved.

import nncore


def test_bind_getters():

    @nncore.bind_getters('name', 'depth')
    class Backbone:
        _name = 'ResNet'
        _depth = 50

    backbone = Backbone()
    assert backbone.name == 'ResNet'
    assert backbone.depth == 50
