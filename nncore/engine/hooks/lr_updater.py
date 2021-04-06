# Copyright (c) Ye Liu. All rights reserved.

from math import cos, pi

import nncore
from .base import Hook
from .builder import HOOKS

POLICIES = nncore.Registry('policy')


@POLICIES.register(name='step')
def _step(base_lr, progress, step, gamma=0.1, **kwargs):
    if isinstance(step, int):
        return base_lr * (gamma**(progress // step))
    exp = len(step)
    for i, s in enumerate(step):
        if progress < s:
            exp = i
            break
    return base_lr * gamma**exp


@POLICIES.register(name='cosine')
def _cosine(base_lr, progress, max_progress, target_lr=0, **kwargs):
    scale = cos(pi * (progress / max_progress)) + 1
    return (base_lr - target_lr) * scale * 0.5 + target_lr


@POLICIES.register(name='exp')
def _exp(base_lr, progress, gamma, **kwargs):
    return base_lr * gamma**progress


@POLICIES.register(name='poly')
def _poly(base_lr, progress, max_progress, power=1, min_lr=0, **kwargs):
    coeff = (1 - progress / max_progress)**power
    return (base_lr - min_lr) * coeff + min_lr


@POLICIES.register(name='inv')
def _inv(base_lr, progress, gamma, power=1, **kwargs):
    return base_lr * (gamma * progress + 1)**(-power)


@HOOKS.register()
class LrUpdaterHook(Hook):
    """
    Update learning rate periodically during training. Currently supported
    learning rate and warm-up policies are ``step``, ``cosine``, ``exp``,
    ``poly``, ``inv`` and ``linear``, ``exp``, ``constant`` respectively.

    Learning rate policy configs:
        - `step`: step (list[int]), gamma (float, Default: ``0.1``)
        - `cosine`: target_lr (float, Default: ``0``)
        - `exp`: gamma (float)
        - `poly`: power (float, Default: ``1``), min_lr (float, Default: ``0``)
        - `inv`: gamma (float), power (float, Default: ``1``)

    Warm-up policy configs:
        - `linear`: ratio (float)
        - `exp`: ratio (float)
        - `constant`: ratio (float)
    """

    def _base_lr(self, engine):
        return [group['base_lr'] for group in engine.optimizer.param_groups]

    def _set_lr(self, engine, lr_groups):
        for group, lr in zip(engine.optimizer.param_groups, lr_groups):
            group['lr'] = lr

    def _update_lr(self, engine, cfg):
        policy = POLICIES.get(self._schd_cfg['policy'])
        lr_groups = [policy(lr, **cfg) for lr in self._base_lr(engine)]
        self._set_lr(engine, lr_groups)
        return lr_groups

    def _warmup_lr(self, lr_groups, progress):
        if self._warm_cfg['policy'] == 'linear':
            scale = (self._warm_cfg['ratio'] - 1) * progress + 1
        elif self._warm_cfg['policy'] == 'exp':
            scale = self._warm_cfg['ratio']**progress
        elif self._warm_cfg['policy'] == 'constant':
            scale = self._warm_cfg['ratio']
        else:
            raise TypeError("invalid warm-up policy: '{}'".format(
                self._warm_cfg['policy']))
        return [lr * scale for lr in lr_groups]

    def before_stage(self, engine):
        self._schd_cfg = engine.cur_stage.get('lr_schedule')
        self._warm_cfg = engine.cur_stage.get('warmup')
        for group in engine.optimizer.param_groups:
            group.setdefault('base_lr', group['lr'])

    def before_train_epoch(self, engine):
        if self._schd_cfg is not None and self._schd_cfg['type'] == 'epoch':
            cfg = self._schd_cfg.copy()
            cfg['progress'] = engine.epoch_in_stage
            cfg['max_progress'] = engine.cur_stage['epochs']
            lr_groups = self._update_lr(engine, cfg)
        else:
            lr_groups = self._base_lr(engine)

        if self._warm_cfg is not None and self._warm_cfg['type'] == 'epoch':
            if engine.epoch_in_stage < self._warm_cfg['steps']:
                progress = 1 - engine.epoch_in_stage / self._warm_cfg['steps']
                lr_groups = self._warmup_lr(lr_groups, progress)
                self._set_lr(engine, lr_groups)
            elif engine.epoch_in_stage == self._warm_cfg['steps']:
                self._set_lr(engine, lr_groups)

    def before_train_iter(self, engine):
        if self._schd_cfg is not None and self._schd_cfg['type'] == 'iter':
            cfg = self._schd_cfg.copy()
            cfg['progress'] = engine.iter_in_stage
            cfg['max_progress'] = engine.cur_stage['epochs'] * len(
                engine.data_loader)
            lr_groups = self._update_lr(engine, cfg)
        else:
            lr_groups = self._base_lr(engine)

        if self._warm_cfg is not None and self._warm_cfg['type'] == 'iter':
            if engine.iter_in_stage < self._warm_cfg['steps']:
                progress = 1 - engine.iter_in_stage / self._warm_cfg['steps']
                lr_groups = self._warmup_lr(lr_groups, progress)
                self._set_lr(engine, lr_groups)
            elif engine.iter_in_stage == self._warm_cfg['steps']:
                self._set_lr(engine, lr_groups)
