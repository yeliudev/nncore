# Copyright (c) Ye Liu. All rights reserved.

from .base import HOOKS, Hook


@HOOKS.register()
class ClosureHook(Hook):

    def __init__(self, name, func):
        super(ClosureHook, self).__init__()
        if isinstance(name, (list, tuple)):
            for n, f in zip(name, func):
                self._add_hook(n, f)
        else:
            self._add_hook(name, func)

    def _add_hook(self, name, func):
        assert hasattr(self, name) and callable(func)
        setattr(self, name, func)
