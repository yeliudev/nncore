# Copyright (c) Ye Liu. All rights reserved.

from .base import Hook
from .builder import HOOKS


@HOOKS.register()
class EvalHook(Hook):
    """
    Perform evaluation periodically during training.

    Args:
        interval (int, optional): The interval of epochs to perform evaluation.
            Default: ``1``.
        run_test (bool, optional): Whether to run the model on the test split
            before performing evaluation. Default: ``False``.
    """

    def __init__(self, interval=1, run_test=False):
        super(EvalHook, self).__init__()
        self._interval = interval
        self._run_test = run_test

    def after_val_epoch(self, engine):
        if not self.every_n_epochs(engine, self._interval) or not hasattr(
                engine.data_loaders['test'].dataset, 'evaluate'):
            return

        if self._run_test:
            self.test_epoch()

        output = engine.evaluate()

        for key in output:
            engine.buffer.update(key, output[key])
