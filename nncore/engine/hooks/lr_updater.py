# Copyright (c) Ye Liu. All rights reserved.

from math import cos, pi

import nncore
from .base import HOOKS, Hook

UPDATERS = nncore.Registry('updaters')


@UPDATERS.register
def _step(base_lr, progress, step, gamma=0.1, **kwargs):
    if isinstance(step, int):
        return base_lr * (gamma**(progress // step))
    exp = len(step)
    for i, s in enumerate(step):
        if progress < s:
            exp = i
            break
    return base_lr * gamma**exp


@UPDATERS.register
def _exp(base_lr, progress, gamma, **kwargs):
    return base_lr * gamma**progress


@UPDATERS.register
def _poly(base_lr, progress, max_progress, power=1.0, min_lr=0.0, **kwargs):
    coeff = (1 - progress / max_progress)**power
    return (base_lr - min_lr) * coeff + min_lr


@UPDATERS.register
def _inv(base_lr, progress, gamma, power=1.0, **kwargs):
    return base_lr * (gamma * progress + 1)**(-power)


@UPDATERS.register
def _cosine(base_lr, progress, max_progress, target_lr=0, **kwargs):
    scale = cos(pi * (progress / max_progress)) + 1
    return (base_lr - target_lr) * scale * 0.5 + target_lr


@HOOKS.register
class LrUpdaterHook(Hook):

    def _update_lr(self, engine, cfg):
        updater = UPDATERS.get('_{}'.format(self._cfg.policy))
        for group in engine.optimizer.param_groups:
            lr = updater(group['initial_lr'], **cfg)
            group['lr'] = group['updated_lr'] = lr

    def before_stage(self, engine):
        self._cfg = engine.cur_stage.getdefault('lr_updater', None)
        for group in engine.optimizer.param_groups:
            group.setdefault('initial_lr', group['lr'])

    def before_train_epoch(self, engine):
        if self._cfg is None or self._cfg.type != 'epoch':
            return

        cfg = self._cfg.copy()
        cfg.progress = engine.epoch_in_stage
        cfg.max_progress = engine.cur_stage.epochs

        self._update_lr(engine, cfg)

    def before_train_iter(self, engine):
        if self._cfg is None or self._cfg.type != 'iter':
            return

        cfg = self._cfg.copy()
        cfg.progress = engine.iter_in_stage
        cfg.max_progress = engine.cur_stage.epochs * len(engine.data_loader)

        self._update_lr(engine, cfg)
