# Copyright (c) Ye Liu. All rights reserved.

import os.path as osp
from collections import OrderedDict
from datetime import timedelta

import torch
import torch.distributed as dist

import nncore
from ..comm import get_world_size, master_only
from .base import HOOKS, Hook

WRITERS = nncore.Registry('writer')


def _collect_metrics(engine, mode, window_size):
    metrics = OrderedDict()

    metrics['mode'] = mode
    metrics['epoch'] = engine.epoch
    metrics['iter'] = engine.iter_in_epoch

    if len(engine.optimizer.param_groups) == 1:
        metrics['lr'] = round(engine.optimizer.param_groups[0]['lr'], 4)
    else:
        metrics['lr'] = [
            round(group['lr'], 4) for group in engine.optimizer.param_groups
        ]

    if mode == 'train':
        metrics['epoch'] += 1
        metrics['iter'] += 1
        metrics['time'] = engine.buffer.mean(
            '_iter_time', window_size=window_size)
        metrics['data_time'] = engine.buffer.mean(
            '_data_time', window_size=window_size)

    return metrics


@WRITERS.register
class CommandLineWriter(object):

    _t_log = 'Epoch [{}][{}/{}] lr: {:.4f}, eta: {}, time: {:.3f}, data_time: {:.3f}, '  # noqa:E501
    _v_log = 'Epoch({}) [{}][{}] '

    def write(self, engine, mode, window_size):
        metrics = _collect_metrics(engine, mode, window_size)

        if mode == 'train':
            total_time = engine.buffer.latest('_total_time')
            num_iter_passed = engine.iter + 1 - engine.start_iter
            num_iter_left = engine.max_iters - engine.iter - 1
            eta = timedelta(
                seconds=int(num_iter_left * total_time / num_iter_passed))

            log = self._t_log.format(metrics['epoch'], metrics['iter'],
                                     len(engine.data_loader), metrics['lr'],
                                     eta, metrics['time'],
                                     metrics['data_time'])

            if torch.cuda.is_available():
                mem = torch.cuda.max_memory_allocated()
                mem_mb = torch.IntTensor([mem / (1024 * 1024)],
                                         device=torch.device('cuda'))
                if get_world_size() > 1:
                    dist.reduce(mem_mb, 0, op=dist.ReduceOp.MAX)
                log += 'memory: {}, '.format(mem_mb.item())
        else:
            log = self._v_log.format(mode, metrics['epoch'],
                                     len(engine.data_loader))

        extra = []
        for key in engine.buffer.keys():
            if not key.startswith('_'):
                extra.append('{}: {:.4f}'.format(
                    key, engine.buffer.mean(key, window_size=window_size)))

        log += ', '.join(extra)
        engine.logger.info(log)


@WRITERS.register
class JSONWriter(object):

    def __init__(self, filename='metrics.json'):
        """
        Args:
            filename (str, optional): name of the output JSON file
        """
        self._filename = filename

    def write(self, engine, mode, window_size):
        metrics = _collect_metrics(engine, mode, window_size)

        for key in engine.buffer.keys():
            if not key.startswith('_') and not key.endswith('_'):
                metrics[key] = engine.buffer.mean(key, window_size=window_size)

        filename = osp.join(engine.work_dir, self._filename)
        with open(filename, 'a+') as f:
            nncore.dump(metrics, f, file_format='json')
            f.write('\n')


@HOOKS.register
class EventWriterHook(Hook):

    def __init__(self, interval=50, writers=[]):
        super(EventWriterHook, self).__init__()
        self._interval = interval
        self._writers = [nncore.build_object(w, WRITERS) for w in writers]

    def _log(self, engine, mode):
        for w in self._writers:
            w.write(engine, mode, self._interval)

    def _empty_buffer(self, engine):
        for key in list(engine.buffer.keys()):
            if not key.startswith('_'):
                engine.buffer.clear(key)

    @master_only
    def after_train_iter(self, engine):
        if not self.last_iter_in_epoch(engine) and not self.every_n_iters(
                engine, self._interval):
            return

        self._log(engine, 'train')
        self._empty_buffer(engine)

    @master_only
    def after_val_epoch(self, engine):
        self._log(engine, 'val')
        self._empty_buffer(engine)
