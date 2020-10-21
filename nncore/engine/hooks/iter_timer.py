# Copyright (c) Ye Liu. All rights reserved.

from datetime import timedelta

from nncore import Timer
from ..comm import master_only
from .base import HOOKS, Hook


@HOOKS.register()
class IterTimerHook(Hook):

    def __init__(self):
        super(IterTimerHook, self).__init__()
        self._total_timer = Timer()
        self._iter_timer = Timer()
        self._data_timer = Timer()
        self._train_timer = Timer()
        self._val_timer = Timer()

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

        num_iter = engine.iter - engine.start_iter

        if num_iter > 0 and train_time > 0:
            engine.logger.info(
                'Overall training speed: {} iterations in {} ({:.4f} s / it)'.
                format(num_iter, timedelta(seconds=int(train_time)),
                       train_time / num_iter))

        engine.logger.info('Done running in {} ({} on hooks)'.format(
            timedelta(seconds=int(total_time)),
            timedelta(seconds=int(hook_time))))

    @master_only
    def before_epoch(self, engine):
        for key in list(engine.buffer.keys()):
            if key in [
                    '_total_time', '_iter_time', '_data_time', '_train_time',
                    '_val_time'
            ]:
                engine.buffer.remove(key)

    @master_only
    def before_train_iter(self, engine):
        data_time = self._data_timer.seconds()
        engine.buffer.update('_data_time', data_time)

        self._iter_timer.reset()
        self._train_timer.resume()

    @master_only
    def after_train_iter(self, engine):
        total_time = self._total_timer.seconds()
        engine.buffer.update('_total_time', total_time)

        step_time = self._iter_timer.seconds()
        engine.buffer.update('_iter_time', step_time)

        train_time = self._train_timer.seconds()
        engine.buffer.update('_train_time', train_time)

        self._data_timer.reset()
        self._train_timer.pause()

    @master_only
    def before_val_iter(self, engine):
        data_time = self._data_timer.seconds()
        engine.buffer.update('_data_time', data_time)

        self._iter_timer.reset()
        self._val_timer.resume()

    @master_only
    def after_val_iter(self, engine):
        total_time = self._total_timer.seconds()
        engine.buffer.update('_total_time', total_time)

        step_time = self._iter_timer.seconds()
        engine.buffer.update('_iter_time', step_time)

        val_time = self._val_timer.seconds()
        engine.buffer.update('_val_time', val_time)

        self._data_timer.reset()
        self._val_timer.pause()
