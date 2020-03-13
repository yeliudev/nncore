# Copyright (c) Ye Liu. All rights reserved.

import nncore


def test_bind_getter():

    @nncore.bind_getter('name', 'depth')
    class Backbone:
        _name = 'ResNet'
        _depth = 50

    backbone = Backbone()
    assert backbone.name == 'ResNet'
    assert backbone.depth == 50
