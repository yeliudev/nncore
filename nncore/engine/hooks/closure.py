# Copyright (c) Ye Liu. All rights reserved.

from .base import HOOKS, Hook


@HOOKS.register()
class ClosureHook(Hook):

    def __init__(self, fn_name, fn):
        if isinstance(fn, (list, tuple)):
            for name, f in zip([fn_name, fn]):
                self._add_hook(name, f)
        else:
            self._add_hook(fn_name, fn)

    def _add_hook(self, fn_name, fn):
        assert hasattr(self, fn_name)
        assert callable(fn)
        setattr(self, fn_name, fn)
