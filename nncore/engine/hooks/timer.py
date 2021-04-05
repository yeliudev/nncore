# Copyright (c) Ye Liu. All rights reserved.

from datetime import timedelta

import nncore
from ..comm import master_only
from .base import Hook
from .builder import HOOKS


@HOOKS.register()
class TimerHook(Hook):
    """
    Compute and save timings into :obj:`enging.buffer` during training.
    """

    def __init__(self):
        super(TimerHook, self).__init__()
        self._total_timer = nncore.Timer()
        self._iter_timer = nncore.Timer()
        self._data_timer = nncore.Timer()
        self._train_timer = nncore.Timer()
        self._val_timer = nncore.Timer()

    def _update_time(self, engine, keys):
        for key in keys:
            engine.buffer.update(
                '_{}_time'.format(key),
                getattr(self, '_{}_timer'.format(key)).seconds())

    @master_only
    def before_launch(self, engine):
        self._total_timer.reset()
        self._data_timer.reset()
        self._train_timer.reset()
        self._train_timer.pause()
        self._val_timer.reset()
        self._val_timer.pause()

    @master_only
    def after_launch(self, engine):
        total_time = self._total_timer.seconds()
        train_time = self._train_timer.seconds()
        val_time = self._val_timer.seconds()

        hook_time = total_time - train_time - val_time
        num_iters = engine.iter - engine.start_iter

        if num_iters > 0 and train_time > 0:
            engine.logger.info(
                'Overall training speed: {} iterations in {} ({:.4f} s / it)'.
                format(num_iters, timedelta(seconds=int(train_time)),
                       train_time / num_iters))

        engine.logger.info('Done running in {} ({} on hooks)'.format(
            timedelta(seconds=int(total_time)),
            timedelta(seconds=int(hook_time))))

    @master_only
    def before_epoch(self, engine):
        for key in list(engine.buffer.keys()):
            if key in ['total', 'iter', 'data', 'train', 'val']:
                engine.buffer.pop('_{}_time'.format(key))

    @master_only
    def before_train_iter(self, engine):
        self._iter_timer.reset()
        self._train_timer.resume()
        self._update_time(engine, ['data'])

    @master_only
    def after_train_iter(self, engine):
        self._data_timer.reset()
        self._train_timer.pause()
        self._update_time(engine, ['total', 'iter', 'train'])

    @master_only
    def before_val_iter(self, engine):
        self._iter_timer.reset()
        self._val_timer.resume()
        self._update_time(engine, ['data'])

    @master_only
    def after_val_iter(self, engine):
        self._data_timer.reset()
        self._val_timer.pause()
        self._update_time(engine, ['total', 'iter', 'val'])
