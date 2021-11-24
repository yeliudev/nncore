# Copyright (c) Ye Liu. All rights reserved.

from ..builder import HOOKS
from .base import Hook


@HOOKS.register()
class EvalHook(Hook):
    """
    Perform evaluation periodically during training.

    Args:
        interval (int, optional): The interval of epochs to perform evaluation.
            Default: ``1``.
        run_test (bool, optional): Whether to run the model on the test split
            before performing evaluation. Default: ``False``.
        high_keys (list[str], optional): The list of metrics (higher is better)
            to be compared. Default: ``[]``.
        low_keys (list[str], optional): The list of metrics (lower is better)
            to be compared. Default: ``[]``.
    """

    def __init__(self, interval=1, run_test=False, high_keys=[], low_keys=[]):
        super(EvalHook, self).__init__()

        self._interval = interval
        self._run_test = run_test
        self._high_keys = high_keys
        self._low_keys = low_keys

        self._high_values = {k: float('-inf') for k in high_keys}
        self._low_values = {k: float('inf') for k in low_keys}

    def after_val_epoch(self, engine):
        if (not self.every_n_epochs(engine, self._interval) or
                not hasattr(engine.data_loaders['test'].dataset, 'evaluate')):
            return

        if self._run_test:
            engine.test_epoch()

        output = engine.evaluate()

        for key, value in output.items():
            engine.buffer.update(key, value)

        for key in self._high_keys:
            if key not in output:
                continue
            if output[key] > self._high_values[key]:
                self._high_values[key] = output[key]
            engine.buffer.update('best_{}'.format(key), self._high_values[key])

        for key in self._low_keys:
            if key not in output:
                continue
            if output[key] < self._low_values[key]:
                self._low_values[key] = output[key]
            engine.buffer.update('best_{}'.format(key), self._low_values[key])
