# Copyright (c) Ye Liu. All rights reserved.

from collections import OrderedDict

import nncore
from .hooks import HOOKS, Hook
from .utils import bind_hooks


@bind_hooks
@nncore.bind_getter('hooks', 'max_stages', 'stage', 'epoch', 'iter')
class Engine(object):

    def __init__(self, model, data_loaders, hooks, logger=None, work_dir=None):
        self.model = model
        self.data_loaders = data_loaders
        self.work_dir = work_dir

        for hook in hooks:
            self.register_hook(hook)

        self.logger = logger or nncore.get_logger()
        self.flush_states()

    def flush_states(self):
        self._hooks = OrderedDict()
        self._max_stages = len(self.stages)
        self._stage = 0
        self._epoch = 0
        self._iter = 0

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

    def train_step(self, *args, **kwargs):
        self.before_train_step()
        # do something
        self.after_train_step()
        self._step += 1

    def val_step(self, *args, **kwargs):
        self.before_val_step()
        # do something
        self.after_val_step()

    def train_epoch(self, *args, **kwargs):
        self.model.train()
        self.before_train_epoch()
        # do something
        self.after_train_epoch()
        self._epoch += 1

    def val_epoch(self, *args, **kwargs):
        self.model.eval()
        self.before_val_epoch()
        # do something
        self.after_val_epoch()

    def train_stage(self, *args, **kwargs):
        self.before_stage()
        self.cur_stage = self._stages[self._stage]
        self.logger.info('Stage {}, num_epochs: {}'.format(
            self._stage, self.cur_stage.epochs))
        for i in range(self.cur_stage.epochs):
            self.train_epoch()
            if i % self.cur_stage.val_interval == 0:
                self.val_epoch()
        self.after_stage()
        self._stage += 1

    def launch(self):
        self.logger.info('Launch engine, host: {}, work_dir: {}'.format(
            nncore.get_host_info(), self.work_dir))
        self.before_launch()
        while self._stage < self._max_stages:
            self.train_stage()
        self.after_launch()
