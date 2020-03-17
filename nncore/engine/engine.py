# Copyright (c) Ye Liu. All rights reserved.

from collections import OrderedDict

import torch

import nncore
from .hooks import HOOKS, Hook
from .utils import bind_hooks


@bind_hooks
@nncore.bind_getter('hooks', 'max_stages', 'max_epochs', 'stage', 'epoch',
                    'iter', 'step')
class Engine(object):

    def __init__(self,
                 model,
                 data_loaders,
                 stages,
                 hooks=None,
                 logger=None,
                 work_dir=None):
        self.model = model
        self.data_loaders = data_loaders
        self.stages = stages
        self.work_dir = work_dir

        self._hooks = OrderedDict()
        if hooks is not None:
            for hook in hooks:
                self.register_hook(hook)

        self.logger = logger or nncore.get_logger()
        self.flush_states()

    @property
    def cur_stage(self):
        return self.stages[self._stage]

    @property
    def period(self):
        cumsum = 0
        for stage in self.stages:
            if self._epoch + 1 <= cumsum + stage.epochs:
                return self._epoch - cumsum
            cumsum += stage.epochs

    def flush_states(self):
        self._max_stages = len(self.stages)
        self._max_epochs = sum(stage.epochs for stage in self.stages)
        self._stage = 0
        self._epoch = 0
        self._iter = 0
        self._step = 0

    def register_hook(self, hook, before=None):
        """
        Register a hook into the engine.

        Args:
            hook (:obj:`Hook` or dict): the hook to be registered
            before (str, optional): name of the hook to be inserted before. The
                new hook will be inserted into the end of the hook list by
                default.
        """
        if isinstance(hook, dict):
            hook = nncore.build_object(hook, HOOKS)
        elif not isinstance(hook, Hook):
            raise TypeError('hook must be a Hook or dict, but got {}'.format(
                type(hook)))

        if hook in self._hooks:
            raise ValueError("hook '{}' exists".format(hook.name))

        hook.on_register(self)
        self._hooks[hook.name] = hook

        if before is not None:
            if before not in self._hooks:
                raise ValueError("hook '{}' not found".format(before))

            keys = list(self._hooks.keys())
            idx = keys.index(before)
            for key in keys[idx:-1]:
                self._hooks[key].move_to_end()

    def build_optimizer(self, optimizer):
        """
        Build an optimizer for the engine.

        Args:
            optimizer (any): an optimizer object or a dict used for
                constructing the optimizer
        """
        if isinstance(optimizer, dict):
            self.optimizer = nncore.build_object(
                optimizer, torch.optim, dict(params=self.model.parameters()))
        elif hasattr(optimizer, 'zero_grad') and hasattr(optimizer, 'step'):
            self.optimizer = optimizer
        else:
            raise TypeError("invalid optimizer: {}".format(optimizer))

    def _train_iter(self, *args, **kwargs):
        self.before_train_iter()
        self.train_iter(*args, **kwargs)
        self.after_train_iter()
        self._iter += 1

    def _val_iter(self, *args, **kwargs):
        self.before_val_iter()
        self.val_iter(*args, **kwargs)
        self.after_val_iter()

    def _train_epoch(self, *args, **kwargs):
        self.model.train()
        self.before_train_epoch()
        self.train_epoch(*args, **kwargs)
        self.after_train_epoch()
        self._epoch += 1

    def _val_epoch(self, *args, **kwargs):
        self.model.eval()
        self.before_val_epoch()
        self.val_epoch(*args, **kwargs)
        self.after_val_epoch()

    def _train_stage(self, *args, **kwargs):
        self.before_stage()
        self.train_stage(*args, **kwargs)
        self.after_stage()
        self._stage += 1

    def train_iter(self, data):
        output = self.model(data, return_loss=True)
        loss = sum(v for k, v in output.items() if 'loss' in k)
        log_vars = {k: v.item() for k, v in output.items()}
        log_vars['loss'] = loss.item()
        self.iter_output = dict(loss=loss, log_vars=log_vars)
        return output

    def val_iter(self, data):
        with torch.no_grad():
            output = self.model(data, return_loss=True)
        loss = sum(v for k, v in output.items() if 'loss' in k)
        log_vars = {k: v.item() for k, v in output.items()}
        log_vars['loss'] = loss.item()
        self.iter_output = dict(log_vars=log_vars)
        return output

    def train_epoch(self):
        collect_output = getattr(self.cur_stage, 'collect_train_output', False)
        self.epoch_output = [] if collect_output else None
        self.data_loader = self.data_loaders['train']
        for step, data in enumerate(self.data_loader):
            self._step = step
            output = self._train_iter(data)
            if collect_output:
                self.epoch_output.append(output)

    def val_epoch(self):
        collect_output = getattr(self.cur_stage, 'collect_val_output', False)
        self.epoch_output = [] if collect_output else None
        self.data_loader = self.data_loaders['val']
        for step, data in enumerate(self.data_loader):
            self._step = step
            output = self._val_iter(data)
            if collect_output:
                self.epoch_output.append(output)

    def train_stage(self):
        if self.period == 0:
            self.build_optimizer(self.cur_stage.optimizer)
        interval = getattr(self.cur_stage, 'val_interval', 0)
        while self.period < self.cur_stage.epochs:
            self._train_epoch()
            if interval > 0 and self.period % interval == 0:
                self._val_epoch()

    def launch(self):
        self.before_launch()
        while self._stage < self._max_stages:
            self._train_stage()
        self.after_launch()
