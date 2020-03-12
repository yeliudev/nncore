# Copyright (c) Ye Liu. All rights reserved.


class Hook(object):
    """
    Base class for hooks that can be registered with :class:`Engine`.

    Each hook can implement several methods. In the hook methods, users should
    provide an argument `engine` to access more properties about the context.
    All the hooks will be called one by one according to the order in
    `engine.hooks`.
    """

    def before_run(self, engine):
        pass

    def after_run(self, engine):
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

    def before_val_epoch(self, engine):
        self.before_epoch(engine)

    def after_train_epoch(self, engine):
        self.after_epoch(engine)

    def after_val_epoch(self, engine):
        self.after_epoch(engine)

    def before_train_iter(self, engine):
        self.before_iter(engine)

    def before_val_iter(self, engine):
        self.before_iter(engine)

    def after_train_iter(self, engine):
        self.after_iter(engine)

    def after_val_iter(self, engine):
        self.after_iter(engine)

    def every_n_inner_iters(self, runner, n):
        return (runner.inner_iter + 1) % n == 0 if n > 0 else False

    def every_n_iters(self, runner, n):
        return (runner.iter + 1) % n == 0 if n > 0 else False

    def every_n_epochs(self, runner, n):
        return (runner.epoch + 1) % n == 0 if n > 0 else False

    def end_of_epoch(self, runner):
        return runner.inner_iter + 1 == len(runner.data_loader)
