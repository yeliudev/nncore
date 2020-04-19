# Copyright (c) Ye Liu. All rights reserved.

from collections import OrderedDict

import torch.distributed as dist
from torch._utils import (_flatten_dense_tensors, _take_tensors,
                          _unflatten_dense_tensors)
from torch.nn.utils import clip_grad

from .base import HOOKS, Hook


def _allreduce_coalesced(tensors, world_size, bucket_size_mb):
    if bucket_size_mb > 0:
        bucket_size_bytes = bucket_size_mb * 1024 * 1024
        buckets = _take_tensors(tensors, bucket_size_bytes)
    else:
        buckets = OrderedDict()
        for tensor in tensors:
            tp = tensor.type()
            if tp not in buckets:
                buckets[tp] = []
            buckets[tp].append(tensor)
        buckets = buckets.values()

    for bucket in buckets:
        flat_tensors = _flatten_dense_tensors(bucket)
        dist.all_reduce(flat_tensors)
        flat_tensors.div_(world_size)
        for tensor, synced in zip(
                bucket, _unflatten_dense_tensors(flat_tensors, bucket)):
            tensor.copy_(synced)


def _allreduce_grads(params, coalesce, bucket_size_mb):
    grads = [
        param.grad.data for param in params
        if param.requires_grad and param.grad is not None
    ]
    world_size = dist.get_world_size()
    if coalesce:
        _allreduce_coalesced(grads, world_size, bucket_size_mb)
    else:
        for tensor in grads:
            dist.all_reduce(tensor.div_(world_size))


@HOOKS.register
class OptimizerHook(Hook):

    def __init__(self, interval=1):
        super(OptimizerHook, self).__init__()
        self._interval = interval

    def _avg_grads(self, engine):
        step_size = engine.iter_in_epoch - self._last_update_iter + 1
        for param in engine.model.parameters():
            if param.requires_grad and param.grad is not None:
                param.grad.data.div_(step_size)
        self._last_update_iter = engine.iter_in_epoch + 1

    def _clip_grads(self, params, cfg):
        params_with_grad = filter(
            lambda p: p.requires_grad and p.grad is not None, params)
        if len(params_with_grad) > 0:
            clip_grad.clip_grad_norm_(params_with_grad, **cfg)

    def before_train_epoch(self, engine):
        self._last_update_iter = 0
        engine.optimizer.zero_grad()

    def after_train_iter(self, engine):
        loss_type = engine.cur_stage.get('loss', 'loss')
        engine.losses[loss_type].backward()

        if self.every_n_iters_in_epoch(
                engine, self._interval) or self.last_iter_in_epoch(engine):
            self._avg_grads(engine)
            grad_clip = engine.cur_stage.get('grad_clip', None)
            if grad_clip is not None:
                self._clip_grads(engine.model.parameters(), grad_clip)
            engine.optimizer.step()
            engine.optimizer.zero_grad()

    def after_train_epoch(self, engine):
        engine.optimizer.zero_grad()


@HOOKS.register
class DistOptimizerHook(OptimizerHook):

    def __init__(self, interval=1, coalesce=True, bucket_size_mb=-1):
        super(DistOptimizerHook, self).__init__()
        self._interval = interval
        self._coalesce = coalesce
        self._bucket_size_mb = bucket_size_mb

    def _avg_grads(self, engine):
        super(DistOptimizerHook, self)._avg_grads(engine)
        _allreduce_grads(engine.model.parameters(), self._coalesce,
                         self._bucket_size_mb)
