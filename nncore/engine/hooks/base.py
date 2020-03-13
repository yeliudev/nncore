# Copyright (c) Ye Liu. All rights reserved.

import nncore


@nncore.bind_getter('name')
class Hook(object):
    """
    Base class for hooks that can be registered into :class:`Engine`.

    Each hook can implement several methods. In the hook methods, users should
    provide an argument `engine` to access more properties about the context.
    All the hooks will be called one by one according to the order in
    `engine.hooks`.
    """

    def __init__(self, name=None):
        self._name = name or self.__class__.__name__

    def __repr__(self):
        return '{}()'.format(self._name)

    def __eq__(self, hook):
        return self._name == getattr(hook, 'name', hook)

    def before_run(self, engine):
        pass

    def after_run(self, engine):
        pass

    def before_stage(self, engine):
        pass

    def after_stage(self, engine):
        pass

    def before_epoch(self, engine):
        pass

    def after_epoch(self, engine):
        pass

    def before_step(self, engine):
        pass

    def after_step(self, engine):
        pass

    def before_train_epoch(self, engine):
        self.before_epoch(engine)

    def after_train_epoch(self, engine):
        self.after_epoch(engine)

    def before_val_epoch(self, engine):
        self.before_epoch(engine)

    def after_val_epoch(self, engine):
        self.after_epoch(engine)

    def before_train_step(self, engine):
        self.before_step(engine)

    def after_train_step(self, engine):
        self.after_step(engine)

    def before_val_step(self, engine):
        self.before_step(engine)

    def after_val_step(self, engine):
        self.after_step(engine)

    def every_n_epochs(self, runner, n):
        return (runner.epoch + 1) % n == 0 if n > 0 else False

    def every_n_steps(self, runner, n):
        return (runner.iter + 1) % n == 0 if n > 0 else False
