# Copyright (c) Ye Liu. All rights reserved.

from torch.nn.modules.batchnorm import _BatchNorm

from nncore.nn import update_bn_stats_
from .base import Hook
from .builder import HOOKS


@HOOKS.register()
class PreciseBNHook(Hook):
    """
    Compute Precise-BN using EMA periodically during training. This hook will
    also run in the end of training.

    Args:
        interval (int, optional): The interval of epochs to compute the stats.
            Default: ``1``.
        num_iters (int, optional): Number of iterations to compute the stats.
            This number will be overwritten by the length of training data
            loader. Default: ``200``.
    """

    def __init__(self, interval=1, num_iters=200):
        super(PreciseBNHook, self).__init__()
        self._interval = interval
        self._num_iters = num_iters

    def after_train_epoch(self, engine):
        if not self.every_n_epochs(
                engine, self._interval) and not self.last_epoch(engine):
            return

        if any(m for m in engine.model.modules()
               if isinstance(m, _BatchNorm) and m.training):
            engine.logger.info('Computing Precise BN...')
            num_iters = min(self._num_iters, len(engine.data_loader))
            update_bn_stats_(
                engine.model,
                engine.data_loader,
                num_iters=num_iters,
                mode=engine.mode,
                **engine.kwargs)
