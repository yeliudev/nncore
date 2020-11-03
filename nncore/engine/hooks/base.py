# Copyright (c) Ye Liu. All rights reserved.

import nncore

HOOKS = nncore.Registry('hook')


@nncore.bind_getter('name')
class Hook(object):
    """
    Base class for hooks that can be registered into :obj:`Engine`.

    Each hook can implement several methods. In the hook methods, users should
    provide an argument `engine` to access more properties about the context.
    All the hooks will be called one by one according to the order in
    `engine.hooks`.
    """

    def __init__(self, name=None):
        self._name = name or self.__class__.__name__

    def __eq__(self, hook):
        return self._name == hook.name

    def __repr__(self):
        return '{}()'.format(self._name)

    def before_launch(self, engine):
        pass

    def after_launch(self, engine):
        pass

    def before_stage(self, engine):
        pass

    def after_stage(self, engine):
        pass

    def before_epoch(self, engine):
        pass

    def after_epoch(self, engine):
        pass

    def before_iter(self, engine):
        pass

    def after_iter(self, engine):
        pass

    def before_train_epoch(self, engine):
        self.before_epoch(engine)

    def after_train_epoch(self, engine):
        self.after_epoch(engine)

    def before_val_epoch(self, engine):
        self.before_epoch(engine)

    def after_val_epoch(self, engine):
        self.after_epoch(engine)

    def before_train_iter(self, engine):
        self.before_iter(engine)

    def after_train_iter(self, engine):
        self.after_iter(engine)

    def before_val_iter(self, engine):
        self.before_iter(engine)

    def after_val_iter(self, engine):
        self.after_iter(engine)

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
