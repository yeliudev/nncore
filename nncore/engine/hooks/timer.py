# Copyright (c) Ye Liu. All rights reserved.

from datetime import timedelta
from time import perf_counter

from nncore import Timer
from .base import HOOKS, Hook


@HOOKS.register
class IterTimerHook(Hook):

    def __init__(self):
        super(IterTimerHook, self).__init__()
        self._start_time = perf_counter()
        self._step_timer = Timer()
        self._train_timer = Timer()
        self._val_timer = Timer()
        self._data_timer = Timer()

    def before_launch(self, engine):
        self._start_time = perf_counter()
        self._train_timer.reset()
        self._train_timer.pause()
        self._val_timer.reset()
        self._val_timer.pause()

    def after_launch(self, engine):
        total_time = perf_counter() - self._start_time
        train_time = self._train_timer.seconds()
        val_time = self._val_timer.seconds()
        hook_time = total_time - train_time - val_time

        num_iter = engine.iter + 1 - engine.start_iter

        if num_iter > 0 and train_time > 0:
            engine.logger.info(
                'Overall training speed: {} iterations in {} ({:.4f} s / it)'.
                format(num_iter, timedelta(seconds=int(train_time)),
                       train_time / num_iter))

        engine.logger.info('Done running in {} ({} on hooks)'.format(
            timedelta(seconds=int(total_time)),
            timedelta(seconds=int(hook_time))))

    def before_train_iter(self, engine):
        data_time = self._data_timer.seconds()
        engine.buffer.update('data_time', data_time)
        self._step_timer.reset()
        self._train_timer.resume()

    def after_train_iter(self, engine):
        time = self._step_timer.seconds()
        engine.buffer.update('time', time)
        self._train_timer.pause()
        self._data_timer.reset()

    def before_val_iter(self, engine):
        self._val_timer.resume()

    def after_val_iter(self, engine):
        self._val_timer.pause()
