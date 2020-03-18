# Copyright (c) Ye Liu. All rights reserved.

import os.path as osp
from collections import OrderedDict

import torch
import torch.distributed as dist

import nncore
from ..comm import get_world_size, master_only
from .base import HOOKS, Hook

WRITERS = nncore.Registry('writer')


@WRITERS.register
class MetricWriter(object):

    _t_log = 'Epoch [{}][{}/{}]\tlr: {:.5f}, time: {:.3f}, data_time: {:.3f}, '
    _v_log = 'Epoch({}) [{}][{}]\t'

    def write(self, engine, info):
        if info['mode'] == 'train':
            log = self._t_log.format(info['epoch'], info['step'],
                                     len(engine.data_loader), info['lr'],
                                     info['time'], info['data_time'])
            if 'memory' in info:
                log += 'memory: {}, '.format(info['memory'])
        else:
            log = self._v_log.format(info['mode'], info['epoch'], info['step'])

        extra = []
        for key in engine.buffer.keys():
            if key not in ['time', 'data_time']:
                extra.append('{}: {:.4f}'.format(key, engine.buffer.avg(key)))

        log += ', '.join(extra)
        engine.logger.info(log)


@WRITERS.register
class JSONWriter(object):

    def __init__(self, filename=None):
        """
        Args:
            filename (str, optional): name of the output JSON file
        """
        self._filename = filename

    def write(self, engine, info):
        if self._filename is None:
            self._filename = osp.join(engine.work_dir, 'metric.json')

        for key in engine.buffer.keys():
            if key not in ['time', 'data_time']:
                info[key] = engine.buffer.avg(key)

        with open(self._filename, 'a+') as f:
            nncore.dump(info, f, file_format='json')
            f.write('\n')


@HOOKS.register
class LoggerHook(Hook):

    def __init__(self, interval=50, writers=[]):
        super(LoggerHook, self).__init__()
        self.interval = interval
        self.writers = [nncore.build_object(w, WRITERS) for w in writers]

    def _log(self, engine, info):
        for w in self.writers:
            w.write(engine, info)

    def _clear_buffer(self, engine):
        for key in ['loss', 'scalar', 'plot', 'time', 'data_time']:
            engine.buffer.clear(key)

    def _get_max_memory(self, engine):
        mem = torch.cuda.max_memory_allocated()
        mem_mb = torch.tensor([mem / (1024 * 1024)],
                              dtype=torch.int,
                              device=torch.device('cuda'))

        if get_world_size() > 1:
            dist.reduce(mem_mb, 0, op=dist.ReduceOp.MAX)

        return mem_mb.item()

    def _collect_basic_info(self, engine, mode):
        info = OrderedDict()

        info['mode'] = mode
        info['stage'] = engine.stage + 1
        info['epoch'] = engine.epoch
        info['period'] = engine.period
        info['iteration'] = engine.iter
        info['step'] = engine.step

        if len(engine.optimizer.param_groups) == 1:
            info['lr'] = engine.optimizer.param_groups[0]['lr']
        else:
            info['lr'] = [g['lr'] for g in engine.optimizer.param_groups]

        if mode == 'train':
            info['epoch'] += 1
            info['period'] += 1
            info['iteration'] += 1

            info['time'] = engine.buffer.avg('time')
            info['data_time'] = engine.buffer.avg('data_time')

            if torch.cuda.is_available():
                info['memory'] = self._get_max_memory(engine)

        return info

    def before_epoch(self, engine):
        self._clear_buffer(engine)

    @master_only
    def after_train_iter(self, engine):
        if not self.every_n_iters(
                engine, self.interval) and not self.end_of_epoch(engine):
            return

        info = self._collect_basic_info(engine, 'train')
        self._log(engine, info)

        self._clear_buffer(engine)

    @master_only
    def after_val_epoch(self, engine):
        info = self._collect_basic_info(engine, 'val')
        self._log(engine, info)
