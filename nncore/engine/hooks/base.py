# Copyright (c) Ye Liu. All rights reserved.

from types import MethodType

import nncore

HOOK_NAMES = [
    'before_launch', 'after_launch', 'before_stage', 'after_stage',
    'before_epoch', 'after_epoch', 'before_iter', 'after_iter',
    'before_train_epoch', 'after_train_epoch', 'before_val_epoch',
    'after_val_epoch', 'before_train_iter', 'after_train_iter',
    'before_val_iter', 'after_val_iter'
]


@nncore.bind_getter('name')
class Hook(object):
    """
    Base class for hooks that can be registered into :obj:`Engine`.

    Each hook can implement several methods. In hook methods, users should
    provide an argument ``engine`` to access more properties about the context.
    All hooks will be called one by one according to the order in
    :obj:`engine.hooks`.
    """

    def __init__(self, name=None):
        self._name = name or self.__class__.__name__

        for hook_name in HOOK_NAMES:
            if hasattr(self, hook_name):
                continue

            token = hook_name.split('_')

            if len(token) == 3:

                def _default_hook(self, engine):
                    getattr(self, '{}_{}'.format(token[0], token[2]))(engine)
            else:

                def _default_hook(self, engine):
                    pass

            setattr(self, hook_name, MethodType(_default_hook, self))

    def __eq__(self, hook):
        return self._name == hook.name

    def __repr__(self):
        return '{}()'.format(self._name)

    def every_n_stages(self, engine, n):
        return (engine.stage + 1) % n == 0 if n > 0 else False

    def every_n_epochs(self, engine, n):
        return (engine.epoch + 1) % n == 0 if n > 0 else False

    def every_n_iters(self, engine, n):
        return (engine.iter + 1) % n == 0 if n > 0 else False

    def every_n_epochs_in_stage(self, engine, n):
        return (engine.epoch_in_stage + 1) % n == 0 if n > 0 else False

    def every_n_iters_in_stage(self, engine, n):
        return (engine.iter_in_stage + 1) % n == 0 if n > 0 else False

    def every_n_iters_in_epoch(self, engine, n):
        return (engine.iter_in_epoch + 1) % n == 0 if n > 0 else False

    def first_epoch_in_stage(self, engine):
        return engine.epoch_in_stage == 0

    def first_iter_in_stage(self, engine):
        return engine.iter_in_stage == 0

    def first_iter_in_epoch(self, engine):
        return engine.iter_in_epoch == 0

    def last_epoch_in_stage(self, engine):
        return engine.epoch_in_stage + 1 == engine.cur_stage['epochs']

    def last_iter_in_stage(self, engine):
        return engine.iter_in_stage + 1 == len(
            engine.data_loaders['train']) * engine.cur_stage['epochs']

    def last_iter_in_epoch(self, engine):
        return engine.iter_in_epoch + 1 == len(engine.data_loaders['train'])

    def last_stage(self, engine):
        return engine.stage + 1 == engine.max_stages

    def last_epoch(self, engine):
        return engine.epoch + 1 == engine.max_epochs

    def last_iter(self, engine):
        return engine.iter + 1 == engine.max_iters
