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


def _collect_basic_info(engine, mode, window_size):
    info = OrderedDict()

    info['mode'] = mode
    info['stage'] = engine.stage + 1
    info['epoch'] = engine.epoch
    info['period'] = engine.period
    info['iteration'] = engine.iter
    info['step'] = engine.step
    info['total_steps'] = len(engine.data_loader)

    if len(engine.optimizer.param_groups) == 1:
        info['lr'] = engine.optimizer.param_groups[0]['lr']
    else:
        info['lr'] = [g['lr'] for g in engine.optimizer.param_groups]

    if mode == 'train':
        info['epoch'] += 1
        info['period'] += 1
        info['iteration'] += 1

        info['time'] = engine.buffer.mean(
            '_step_time', window_size=window_size)
        info['data_time'] = engine.buffer.mean(
            '_data_time', window_size=window_size)

        if torch.cuda.is_available():
            mem = torch.cuda.max_memory_allocated()
            mem_mb = torch.IntTensor([mem / (1024 * 1024)],
                                     device=torch.device('cuda'))

            if get_world_size() > 1:
                dist.reduce(mem_mb, 0, op=dist.ReduceOp.MAX)

            info['memory'] = mem_mb.item()

    return info


@WRITERS.register
class CommandLineWriter(object):

    _t_log = 'Epoch [{}][{}/{}]\tlr: {:.5f}, eta: {}, time: {:.3f}, data_time: {:.3f}, '  # noqa:E501
    _v_log = 'Epoch({}) [{}][{}]\t, '

    def write(self, engine, mode, window_size):
        info = _collect_basic_info(engine, mode, window_size)

        if info['mode'] == 'train':
            total_time = engine.buffer.latest('_total_time')
            num_iter_passed = engine.iter + 1 - engine.start_iter
            num_iter_left = engine.max_iters - engine.iter - 1
            eta = timedelta(
                seconds=int(num_iter_left * total_time / num_iter_passed))

            log = self._t_log.format(info['epoch'], info['step'],
                                     info['total_steps'], info['lr'], eta,
                                     info['time'], info['data_time'])

            if 'memory' in info:
                log += 'memory: {}, '.format(info['memory'])
        else:
            log = self._v_log.format(info['mode'], info['epoch'],
                                     info['total_steps'])

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
        info = _collect_basic_info(engine, mode, window_size)

        for key in engine.buffer.keys():
            if not key.startswith('_') and not key.endswith('_'):
                info[key] = engine.buffer.mean(key, window_size=window_size)

        filename = osp.join(engine.work_dir, self._filename)
        with open(filename, 'a+') as f:
            nncore.dump(info, f, file_format='json')
            f.write('\n')


@HOOKS.register
class LoggerHook(Hook):

    def __init__(self, interval=50, writers=[]):
        super(LoggerHook, self).__init__()
        self.interval = interval
        self.writers = [nncore.build_object(w, WRITERS) for w in writers]

    def _log(self, engine, mode):
        for w in self.writers:
            w.write(engine, mode, self.interval)

    def _empty_buffer(self, engine):
        for key in list(engine.buffer.keys()):
            if not key.startswith('_'):
                engine.buffer.clear(key)

    @master_only
    def after_train_iter(self, engine):
        if not self.every_n_iters(
                engine, self.interval) and not self.end_of_epoch(engine):
            return

        self._log(engine, 'train')
        self._empty_buffer(engine)

    @master_only
    def after_val_epoch(self, engine):
        self._log(engine, 'val')
        self._empty_buffer(engine)
