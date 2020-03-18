# Copyright (c) Ye Liu. All rights reserved.

from collections import OrderedDict

import torch

import nncore
from .buffer import Buffer
from .hooks import HOOKS, Hook


@nncore.bind_getter('hooks', 'max_stages', 'max_epochs', 'start_iter', 'stage',
                    'epoch', 'iter', 'step')
class Engine(object):

    def __init__(self,
                 model,
                 data_loaders,
                 stages,
                 buffer_size=990125,
                 hooks=None,
                 logger=None,
                 work_dir=None):
        self.model = model
        self.data_loaders = data_loaders
        self.stages = stages
        self.work_dir = work_dir

        if work_dir is not None:
            nncore.mkdir(work_dir)

        self._hooks = OrderedDict()
        if hooks is not None:
            for hook in hooks:
                self.register_hook(hook)

        self.buffer = Buffer(max_size=buffer_size)
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
        return self.stages[-1].epochs

    def flush_states(self):
        self._max_stages = len(self.stages)
        self._max_epochs = sum(stage.epochs for stage in self.stages)
        self._start_iter = 0
        self._stage = 0
        self._epoch = 0
        self._iter = 0
        self._step = 0

    def _call_hook(self, name):
        for hook in self._hooks.values():
            getattr(hook, name)(self)

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

        if hook.name in self._hooks:
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

    def train_iter(self, data):
        self._call_hook('before_train_iter')

        output = self.model(data, return_loss=True)

        self.losses = {k: v for k, v in output.items() if 'loss' in k}
        self.losses['loss'] = sum(v for v in self.losses.values())

        for key in output:
            self.buffer.update(key, output[key])

        self._call_hook('after_train_iter')
        self._iter += 1

    def val_iter(self, data):
        self._call_hook('before_val_iter')

        with torch.no_grad():
            output = self.model(data, return_loss=True)

        for key in output:
            self.buffer.update(key, output[key])

        self._call_hook('after_val_iter')

    def train_epoch(self):
        self.model.train()
        self._call_hook('before_train_epoch')

        self.data_loader = self.data_loaders['train']
        for step, data in enumerate(self.data_loader):
            self._step = step
            self.train_iter(data)

        self._call_hook('after_train_epoch')
        self._epoch += 1

    def val_epoch(self):
        self.model.eval()
        self._call_hook('before_val_epoch')

        self.data_loader = self.data_loaders['val']
        for step, data in enumerate(self.data_loader):
            self._step = step
            self.val_iter(data)

        self._call_hook('after_val_epoch')

    def run_stage(self):
        self._call_hook('before_stage')
        if self.period == 0:
            self.build_optimizer(self.cur_stage.optimizer)

        n = getattr(self.cur_stage, 'val_interval', 0)
        while self.period < self.cur_stage.epochs:
            self.train_epoch()
            if n > 0 and self.period % n == 0:
                self.val_epoch()

        self._call_hook('after_stage')
        self._stage += 1

    def launch(self):
        self.logger.info('Start running, host: {}, work_dir: {}'.format(
            nncore.get_host_info(), self.work_dir))
        self._call_hook('before_launch')

        while self._stage < self._max_stages:
            self.run_stage()

        self._call_hook('after_launch')
