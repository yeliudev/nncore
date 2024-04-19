# Copyright (c) Ye Liu. Licensed under the MIT License.

from collections import OrderedDict

import torch
import torch.distributed as dist
from torch._utils import (_flatten_dense_tensors, _take_tensors,
                          _unflatten_dense_tensors)
from torch.cuda.amp.grad_scaler import GradScaler
from torch.nn.utils import clip_grad

from ..builder import HOOKS
from ..comm import is_distributed
from .base import Hook


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
        grad_scale (dict | bool | None, optional): Whether to scale the
            gradients. If not specified, this module will automatically scale
            the gradients when amp is activated. Default: ``None``.
    """

    def __init__(self,
                 interval=1,
                 coalesce=True,
                 bucket_size_mb=-1,
                 grad_scale=None):
        super(OptimizerHook, self).__init__()
        self._interval = interval
        self._coalesce = coalesce
        self._bucket_size_mb = bucket_size_mb

        if isinstance(grad_scale, dict):
            grad_scale.setdefault('enabled', True)
            self._grad_scale_cfg = grad_scale
        else:
            self._grad_scale_cfg = dict(enabled=grad_scale)

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

    def before_launch(self, engine):
        cfg = self._grad_scale_cfg.copy()
        enabled = cfg.pop('enabled')
        self.scaler = GradScaler(
            enabled=(engine.get_amp_type() is not None
                     and torch.cuda.is_available())
            if enabled is None else enabled,
            **cfg)

    def before_train_epoch(self, engine):
        self._last_updated_iter = 0
        engine.optimizer.zero_grad()

    def after_train_iter(self, engine):
        key = engine.cur_stage.get('loss', 'loss')
        self.scaler.scale(engine.losses[key]).backward()

        if (not self.every_n_iters_in_epoch(engine, self._interval)
                and not self.last_iter_in_epoch(engine)):
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
            self.scaler.unscale_(engine.optimizer)
            params_with_grad = [
                p for p in engine.model.parameters()
                if p.requires_grad and p.grad is not None
            ]
            if len(params_with_grad) > 0:
                grad_norm = clip_grad.clip_grad_norm_(params_with_grad, **cfg)
                engine.buffer.update('grad_norm', grad_norm.item())

        if engine.debug:
            for name, param in engine.model.named_parameters():
                if param.grad is None:
                    continue
                if param.grad.is_sparse:
                    if param.grad.dtype in (torch.float16, torch.bfloat16):
                        param.grad = param.grad.coalesce()
                    grad = param.grad._values().abs().max()
                else:
                    grad = param.grad.abs().max()
                state = 'Inf' if torch.isinf(grad) else 'NaN' if torch.isnan(
                    grad) else None
                if state is not None:
                    engine.logger.warn('Iter [{}]: {} detected in {}'.format(
                        engine.iter + 1, state, name))

        if self.scaler.is_enabled():
            engine.buffer.update('scale', self.scaler.get_scale())

        self.scaler.step(engine.optimizer)
        self.scaler.update()

        engine.optimizer.zero_grad()

    def after_train_epoch(self, engine):
        engine.optimizer.zero_grad()
