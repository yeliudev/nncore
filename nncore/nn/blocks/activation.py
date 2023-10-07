# Copyright (c) Ye Liu. Licensed under the MIT License.

import torch.nn as nn

import nncore
from ..builder import ACTIVATIONS


@ACTIVATIONS.register()
@nncore.bind_getter('min', 'max')
class Clamp(nn.Module):
    """
    Clamp activation layer.

    Args:
        min (float | None, optional): The lower-bound of the range. Default:
            ``None``.
        max (float | None, optional): The upper-bound of the range. Default:
            ``None``.
    """

    def __init__(self, min=None, max=None):
        super(Clamp, self).__init__()
        self._min = min
        self._max = max

    def forward(self, x):
        return x.clamp(min=self._min, max=self._max)
