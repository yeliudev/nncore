# Copyright (c) Ye Liu. All rights reserved.

from .base import Hook
from .builder import HOOKS


@HOOKS.register()
class ClosureHook(Hook):
    """
    Customize the hooks using self-defined functions.

    Args:
        name (list[str] or str): Name or a list of names of the hooks. Expected
            values include ``'before_launch'``, ``'after_launch'``,
            ``'before_stage'``, ``'after_stage'``, ``'before_epoch'``,
            ``'after_epoch'``, ``'before_iter'``, ``'after_iter'``,
            ``'before_train_epoch'``, ``'after_train_epoch'``,
            ``'before_val_epoch'``, ``'after_val_epoch'``,
            ``'before_train_iter'``, ``'after_train_iter'``,
            ``'before_val_iter'`` and ``'after_val_iter'``
        func (list[function] or function): A function of a list of functions
            for the hooks. These functions should receive an argument
            ``engine`` to access more properties about the context.
    """

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
