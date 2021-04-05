# Copyright (c) Ye Liu. All rights reserved.

from collections import OrderedDict

import torch.distributed as dist
from torch._utils import (_flatten_dense_tensors, _take_tensors,
                          _unflatten_dense_tensors)
from torch.nn.utils import clip_grad

from ..comm import is_distributed
from .base import Hook
from .builder import HOOKS


@HOOKS.register()
class OptimizerHook(Hook):
    """
    Perform back propagation and update parameters of the model periodically.
    This hook supports CPU, single GPU and distributed training.

    Args:
        interval (int, optional): The interval of iterations to update
            parameters. Default: ``1``.
        coalesce (bool, optional): Whether to coalesce the weights in
            distributed training. Default: ``True``.
        bucket_size_mb (int, optional): Size of the bucket. ``-1`` means not
            restricting the bucket size. Default: ``-1``.
    """

    def __init__(self, interval=1, coalesce=True, bucket_size_mb=-1):
        super(OptimizerHook, self).__init__()
        self._interval = interval
        self._coalesce = coalesce
        self._bucket_size_mb = bucket_size_mb

    def _allreduce_coalesced(self, tensors, world_size):
        if self._bucket_size_mb > 0:
            bucket_size_bytes = self._bucket_size_mb * 1024 * 1024
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

    def _allreduce_grads(self, params):
        grads = [
            param.grad.data for param in params
            if param.requires_grad and param.grad is not None
        ]
        world_size = dist.get_world_size()
        if self._coalesce:
            self._allreduce_coalesced(grads, world_size)
        else:
            for tensor in grads:
                dist.all_reduce(tensor.div_(world_size))

    def before_train_epoch(self, engine):
        self._last_updated_iter = 0
        engine.optimizer.zero_grad()

    def after_train_iter(self, engine):
        loss_type = engine.cur_stage.get('loss', 'loss')
        engine.losses[loss_type].backward()

        if not self.every_n_iters_in_epoch(
                engine,
                self._interval) and not self.last_iter_in_epoch(engine):
            return

        step_size = engine.iter_in_epoch - self._last_updated_iter + 1
        for param in engine.model.parameters():
            if param.requires_grad and param.grad is not None:
                param.grad.data.div_(step_size)
        self._last_updated_iter = engine.iter_in_epoch + 1

        if is_distributed():
            self._allreduce_grads(engine.model.parameters())

        cfg = engine.cur_stage.get('grad_clip')
        if cfg is not None:
            params_with_grad = filter(
                lambda p: p.requires_grad and p.grad is not None,
                engine.model.parameters())
            if len(params_with_grad) > 0:
                clip_grad.clip_grad_norm_(params_with_grad, **cfg)

        engine.optimizer.step()
        engine.optimizer.zero_grad()

    def after_train_epoch(self, engine):
        engine.optimizer.zero_grad()
