# Copyright (c) Ye Liu. All rights reserved.

from types import MethodType

import torch

from .base import HOOK_NAMES, Hook
from .builder import HOOKS


@HOOKS.register()
class EmptyCacheHook(Hook):
    """
    Empty cache periodically during training.

    Args:
        names (list[str]): The list of hook names to empty cache. Expected
            values include ``'before_launch'``, ``'after_launch'``,
            ``'before_stage'``, ``'after_stage'``, ``'before_epoch'``,
            ``'after_epoch'``, ``'before_iter'``, ``'after_iter'``,
            ``'before_train_epoch'``, ``'after_train_epoch'``,
            ``'before_val_epoch'``, ``'after_val_epoch'``,
            ``'before_train_iter'``, ``'after_train_iter'``,
            ``'before_val_iter'`` and ``'after_val_iter'``
    """

    def __init__(self, names=[]):
        super(EmptyCacheHook, self).__init__()

        def _empty_cache(self, engine):
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        for name in names:
            assert name in HOOK_NAMES
            setattr(self, name, MethodType(_empty_cache, self))
