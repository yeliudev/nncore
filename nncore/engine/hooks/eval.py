# Copyright (c) Ye Liu. All rights reserved.

from .base import HOOKS, Hook


@HOOKS.register()
class EvalHook(Hook):
    """
    Perform evaluation every specified epoch during training.

    Args:
        interval (int, optional): The interval of performing evaluation.
            Default: ``1``.
        run_test (bool, optional): Whether to run test before performing
            evaluation. Default: ``False``.
    """

    def __init__(self, interval=1, run_test=False):
        super(EvalHook, self).__init__()
        self._interval = interval
        self._run_test = run_test

    def after_val_epoch(self, engine):
        if not self.every_n_epochs(engine, self._interval):
            return

        if self._run_test:
            self.test_epoch()

        output = engine.evaluate()

        for key in output:
            engine.buffer.update(key, output[key])
