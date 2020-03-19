# Copyright (c) Ye Liu. All rights reserved.

from math import cos, pi

import nncore
from .base import HOOKS, Hook

POLICIES = nncore.Registry('policies')


@POLICIES.register
def fixed(base_lr, **kwargs):
    return base_lr


@POLICIES.register
def step(engine, base_lr, type, step, gamma=0.1):
    prog = engine.period + 1 if type == 'epoch' else engine.step + 1

    if isinstance(step, int):
        return base_lr * (gamma**(prog // step))

    exp = len(step)
    for i, s in enumerate(step):
        if prog < s:
            exp = i
            break

    return base_lr * gamma**exp


@POLICIES.register
def exp(engine, base_lr, type, gamma):
    prog = engine.period if type == 'epoch' else engine.step
    return base_lr * gamma**prog


@POLICIES.register
def poly(engine, base_lr, type, power=1.0, min_lr=0.0):
    if type == 'epoch':
        prog = engine.period
        max_prog = engine.cur_stage.epochs
    else:
        prog = engine.step
        max_prog = engine.cur_stage.epochs * len(engine.data_loader)

    coeff = (1 - prog / max_prog)**power
    return (base_lr - min_lr) * coeff + min_lr


@POLICIES.register
def inv(engine, base_lr, type, gamma, power=1.0):
    prog = engine.period if type == 'epoch' else engine.step
    return base_lr * (1 + gamma * prog)**(-power)


@POLICIES.register
def cosine(engine, base_lr, type, target_lr=0):
    if type == 'epoch':
        prog = engine.period
        max_prog = engine.cur_stage.epochs
    else:
        prog = engine.step
        max_prog = engine.cur_stage.epochs * len(engine.data_loader)

    coeff = (1 + cos(pi * (prog / max_prog))) * (base_lr - target_lr)
    return coeff * 0.5 + target_lr


@HOOKS.register
class LrUpdaterHook(Hook):

    def _set_lr(self, engine, lr_groups):
        for group, lr in zip(engine.optimizer.param_groups, lr_groups):
            group['lr'] = lr

    def _get_normal_lr(self, engine):
        return [
            POLICIES.get(self._policy)(engine, lr, self._type, **self._args)
            for lr in self._base_lr
        ]

    def _get_warmup_lr(self, engine):
        normal_lr = self._get_normal_lr(engine)
        n = 1 - engine.step / self._warmup_iters

        if self._warmup == 'linear':
            k = self._warmup_ratio * n - n + 1
        elif self._warmup == 'exp':
            k = self._warmup_ratio**n
        elif self._warmup == 'constant':
            k = self._warmup_ratio

        return [lr * k for lr in normal_lr]

    def before_stage(self, engine):
        self._base_lr = []
        for group in engine.optimizer.param_groups:
            group.setdefault('initial_lr', group['lr'])
            self._base_lr.append(group['initial_lr'])

        _cfg = engine.cur_stage.lr_updater.copy()
        self._policy = _cfg.pop('policy')
        self._type = _cfg.pop('type')
        self._warmup = _cfg.pop('warmup')
        self._warmup_iters = _cfg.pop('warmup_iters')
        self._warmup_ratio = _cfg.pop('warmup_ratio')
        self._args = _cfg

    def before_train_iter(self, engine):
        normal_lr = self._get_normal_lr(engine)
        warmup_lr = self._get_warmup_lr(engine)

        cur_iter = len(engine.data_loader) * engine.period + engine.step + 1

        if self._warmup is None or cur_iter > self._warmup_iters:
            if self._type == 'epoch' and engine.step == 0:
                self._set_lr(engine, normal_lr)
            elif self._type == 'iter':
                self._set_lr(engine, normal_lr)
        elif cur_iter == self._warmup_iters:
            self._set_lr(engine, normal_lr)
        else:
            self._set_lr(engine, warmup_lr)
