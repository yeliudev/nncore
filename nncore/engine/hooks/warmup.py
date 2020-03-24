# Copyright (c) Ye Liu. All rights reserved.

from .base import HOOKS, Hook


@HOOKS.register
class WarmupHook(Hook):

    def _set_warmup_lr(self, engine, progress):
        if self._cfg.policy == 'linear':
            scale = (self._cfg.ratio - 1) * progress + 1
        elif self._cfg.policy == 'exp':
            scale = self._cfg.ratio**progress
        elif self._cfg.policy == 'constant':
            scale = self._cfg.ratio
        else:
            raise TypeError("invalid warmup policy: '{}'".format(
                self._cfg.policy))

        for group in engine.optimizer.param_groups:
            group['lr'] = group['warmup_lr'] = group['updated_lr'] * scale

    def before_stage(self, engine):
        self._cfg = engine.cur_stage.get('warmup', None)
        for group in engine.optimizer.param_groups:
            group.setdefault('updated_lr', group['lr'])

    def before_train_epoch(self, engine):
        if self._cfg is None or self._cfg.type != 'epoch':
            return

        if engine.epoch_in_stage < self._cfg.steps:
            progress = 1 - engine.epoch_in_stage / self._cfg.steps
            self._set_warmup_lr(engine, progress)
        if engine.epoch_in_stage == self._cfg.steps:
            for group in engine.optimizer.param_groups:
                group['lr'] = group['updated_lr']

    def before_train_iter(self, engine):
        if self._cfg is None or self._cfg.type != 'iter':
            return

        if engine.iter_in_stage < self._cfg.steps:
            progress = 1 - engine.iter_in_stage / self._cfg.steps
            self._set_warmup_lr(engine, progress)
        elif engine.iter_in_stage == self._cfg.steps:
            for group in engine.optimizer.param_groups:
                group['lr'] = group['updated_lr']
