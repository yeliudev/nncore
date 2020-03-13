# Copyright (c) Ye Liu. All rights reserved.

from functools import partial

_HOOKS = [
    'before_run', 'after_run', 'before_stage', 'after_stage',
    'before_train_epoch', 'after_train_epoch', 'before_val_epoch',
    'after_val_epoch', 'before_train_step', 'after_train_step',
    'before_val_step', 'after_val_step'
]


def bind_hooks(cls):

    def _call_hook(self, fn_name):
        for hook in self._hooks:
            getattr(hook, fn_name)(self)

    for hook in _HOOKS:
        setattr(cls, hook, partial(_call_hook, fn_name=hook))

    return cls
