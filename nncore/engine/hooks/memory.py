# Copyright (c) Ye Liu. All rights reserved.

from types import MethodType

import torch

from .base import HOOK_NAMES, HOOKS, Hook


@HOOKS.register()
class EmptyCacheHook(Hook):

    def __init__(self, names=[]):
        super(EmptyCacheHook, self).__init__()

        def _empty_cache(self, engine):
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        for name in names:
            assert name in HOOK_NAMES
            setattr(self, name, MethodType(_empty_cache, self))
