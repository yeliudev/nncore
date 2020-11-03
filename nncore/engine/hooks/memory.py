# Copyright (c) Ye Liu. All rights reserved.

from types import MethodType

import torch

from .base import HOOKS, Hook

_PERIODS = [
    'before_launch', 'after_launch', 'before_stage', 'after_stage',
    'before_epoch', 'after_epoch', 'before_iter', 'after_iter',
    'before_train_epoch', 'after_train_epoch', 'before_val_epoch',
    'after_val_epoch', 'before_train_iter', 'after_train_iter',
    'before_val_iter', 'after_val_iter'
]


@HOOKS.register()
class EmptyCacheHook(Hook):

    def __init__(self, periods=[]):
        super(EmptyCacheHook, self).__init__()

        def _empty_cache(self, engine):
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        for period in periods:
            assert period in _PERIODS
            setattr(self, period, MethodType(_empty_cache, self))
