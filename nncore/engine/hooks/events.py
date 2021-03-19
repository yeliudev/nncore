# Copyright (c) Ye Liu. All rights reserved.

from abc import ABCMeta, abstractmethod
from collections import OrderedDict
from datetime import timedelta

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

import nncore
from ..comm import get_world_size, master_only
from .base import HOOKS, Hook

WRITERS = nncore.Registry('writer')


class Writer(metaclass=ABCMeta):

    @abstractmethod
    def write(self, engine, window_size):
        pass

    def open(self, engine):
        pass

    def close(self, engine):
        pass

    def collect_metrics(self, engine, window_size):
        metrics = OrderedDict()

        metrics['mode'] = engine.mode
        metrics['epoch'] = engine.epoch
        metrics['iter'] = engine.iter_in_epoch

        if len(engine.optimizer.param_groups) == 1:
            metrics['lr'] = round(engine.optimizer.param_groups[0]['lr'], 5)
        else:
            metrics['lr'] = [
                round(group['lr'], 5)
                for group in engine.optimizer.param_groups
            ]

        if engine.mode == 'train':
            metrics['epoch'] += 1
            metrics['iter'] += 1
            if '_iter_time' in engine.buffer.keys():
                metrics['time'] = engine.buffer.mean(
                    '_iter_time', window_size=window_size)
            if '_data_time' in engine.buffer.keys():
                metrics['data_time'] = engine.buffer.mean(
                    '_data_time', window_size=window_size)

        return metrics


@WRITERS.register()
class CommandLineWriter(Writer):

    def write(self, engine, window_size):
        metrics = self.collect_metrics(engine, window_size)

        if engine.mode == 'train':
            log = 'Epoch [{}][{}/{}] lr: {:.5f}, '.format(
                metrics['epoch'], metrics['iter'], len(engine.data_loader),
                metrics['lr'])

            if '_total_time' in engine.buffer.keys():
                total_time = engine.buffer.latest('_total_time')
                num_iter_passed = engine.iter + 1 - engine.start_iter
                num_iter_left = engine.max_iters - engine.iter - 1
                eta = timedelta(
                    seconds=int(num_iter_left * total_time / num_iter_passed))
                log += 'eta: {}, '.format(eta)

            for key in ['time', 'data_time']:
                if key in metrics:
                    log += '{}: {:.3f}, '.format(key, metrics[key])

            if next(engine.model.parameters()).device != torch.device('cpu'):
                mem = torch.cuda.max_memory_allocated()
                mem_mb = torch.IntTensor([mem / (1024 * 1024)]).cuda()
                if get_world_size() > 1:
                    dist.reduce(mem_mb, 0, op=dist.ReduceOp.MAX)
                log += 'memory: {}, '.format(mem_mb.item())
        else:
            log = 'Epoch({}) [{}][{}] '.format(engine.mode, metrics['epoch'],
                                               len(engine.data_loader))

        ext = []
        for key in engine.buffer.keys():
            if key.startswith('_') or key.endswith('_'):
                continue

            if isinstance(engine.buffer.latest(key), dict):
                data = engine.buffer.avg(key, window_size=window_size)
                for k, v in data.items():
                    ext.append('{}: {:.4f}'.format('{}_{}'.format(key, k), v))
            else:
                ext.append('{}: {:.4f}'.format(
                    key, engine.buffer.avg(key, window_size=window_size)))

        log += ', '.join(ext)
        engine.logger.info(log)


@WRITERS.register()
class JSONWriter(Writer):

    def __init__(self, filename='metrics.json'):
        """
        Args:
            filename (str, optional): name of the output JSON file
        """
        self._filename = filename

    def open(self, engine):
        nncore.mkdir(engine.work_dir)

    def write(self, engine, window_size):
        metrics = self.collect_metrics(engine, window_size)

        for key in engine.buffer.keys():
            if key.startswith('_') or key.endswith('_'):
                continue

            if isinstance(engine.buffer.latest(key), dict):
                data = engine.buffer.avg(key, window_size=window_size)
                for k, v in data.items():
                    metrics['{}_{}'.format(key, k)] = v
            else:
                metrics[key] = engine.buffer.avg(key, window_size=window_size)

        filename = nncore.join(engine.work_dir, self._filename)
        with open(filename, 'a+') as f:
            nncore.dump(metrics, f, format='json')
            f.write('\n')


@WRITERS.register()
class TensorboardWriter(Writer):

    def __init__(self, log_dir=None, input_to_model=None, **kwargs):
        """
        Args:
            log_dir (str, optional): directory of the tensorboard logs
            input_to_model (Any, optional): the input data or data_loader or
                name of the data_loader for constructing the model graph. If
                None, the graph will not be added.
                Check :meth:`torch.utils.tensorboard.SummaryWriter.add_graph`
                for details about adding a graph to tensorboard.
        """
        self._log_dir = log_dir
        self._input_to_model = input_to_model
        self._kwargs = kwargs

    def open(self, engine):
        if self._log_dir is None:
            self._log_dir = nncore.join(engine.work_dir, 'tf_logs')
        nncore.mkdir(self._log_dir)

        from torch.utils.tensorboard import SummaryWriter
        self._writer = SummaryWriter(self._log_dir, **self._kwargs)

        if self._input_to_model is not None:
            if isinstance(self._input_to_model, DataLoader):
                data = next(iter(self._input_to_model))
            elif isinstance(self._input_to_model, str):
                data = next(iter(engine.data_loaders[self._input_to_model]))
            else:
                data = self._input_to_model

            self._writer.add_graph(engine.model, input_to_model=data)

    def close(self, engine):
        self._writer.close()

    def write(self, engine, window_size):
        for key in engine.buffer.keys():
            if key.startswith('_'):
                continue

            if key.endswith('_'):
                tokens = key.split('_')
                log_type = tokens[-2]

                if log_type not in [
                        'histogram', 'image', 'images', 'figure', 'video',
                        'audio', 'text'
                ]:
                    raise TypeError(
                        "unsupported log type: '{}'".format(log_type))

                tag = '{}/{}'.format(''.join(tokens[:-2]), engine.mode)
                record = engine.buffer.latest(key)
                add_func = getattr(self._writer, 'add_{}'.format(log_type))
                add_func(tag, record, global_step=engine.iter)
            else:
                tag = '{}/{}'.format(key, engine.mode)
                record = engine.buffer.avg(key, window_size=window_size)

                if isinstance(record, dict):
                    self._writer.add_scalars(
                        tag, record, global_step=engine.iter)
                else:
                    self._writer.add_scalar(
                        tag, record, global_step=engine.iter)


@HOOKS.register()
class EventWriterHook(Hook):

    def __init__(self, interval=50, writers=['CommandLineWriter']):
        super(EventWriterHook, self).__init__()
        self._interval = interval
        self._writers = [
            nncore.build_object(w, WRITERS)
            if isinstance(w, dict) else WRITERS.get(w)() for w in writers
        ]

    def _write(self, engine, window_size):
        for w in self._writers:
            w.write(engine, window_size)

    def _empty_buffer(self, engine):
        for key in list(engine.buffer.keys()):
            if not key.startswith('_'):
                engine.buffer.pop(key)

    @master_only
    def before_launch(self, engine):
        for w in self._writers:
            w.open(engine)

    @master_only
    def after_launch(self, engine):
        for w in self._writers:
            w.close(engine)

    @master_only
    def after_train_iter(self, engine):
        if not self.every_n_iters_in_epoch(
                engine,
                self._interval) and not self.last_iter_in_epoch(engine):
            return

        self._write(
            engine,
            len(engine.data_loaders['train']) % self._interval
            or self._interval
            if self.last_iter_in_epoch(engine) else self._interval)
        self._empty_buffer(engine)

    @master_only
    def after_val_epoch(self, engine):
        self._write(engine, len(engine.data_loaders['val']))
        self._empty_buffer(engine)
