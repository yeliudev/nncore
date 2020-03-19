# Copyright (c) Ye Liu. All rights reserved.

import torch

from .base import HOOKS, Hook


@HOOKS.register
class EmptyCacheHook(Hook):

    def __init__(self, timings=[]):
        self._timings = timings

    def before_launch(self, engine):
        if torch.cuda.is_available() and 'before_launch' in self._timings:
            torch.cuda.empty_cache()

    def after_launch(self, engine):
        if torch.cuda.is_available() and 'after_launch' in self._timings:
            torch.cuda.empty_cache()

    def before_stage(self, engine):
        if torch.cuda.is_available() and 'before_stage' in self._timings:
            torch.cuda.empty_cache()

    def after_stage(self, engine):
        if torch.cuda.is_available() and 'after_stage' in self._timings:
            torch.cuda.empty_cache()

    def before_epoch(self, engine):
        if torch.cuda.is_available() and 'before_epoch' in self._timings:
            torch.cuda.empty_cache()

    def after_epoch(self, engine):
        if torch.cuda.is_available() and 'after_epoch' in self._timings:
            torch.cuda.empty_cache()

    def before_iter(self, engine):
        if torch.cuda.is_available() and 'before_iter' in self._timings:
            torch.cuda.empty_cache()

    def after_iter(self, engine):
        if torch.cuda.is_available() and 'after_iter' in self._timings:
            torch.cuda.empty_cache()

    def before_train_epoch(self, engine):
        self.before_epoch(engine)
        if torch.cuda.is_available() and 'before_train_epoch' in self._timings:
            torch.cuda.empty_cache()

    def after_train_epoch(self, engine):
        self.after_epoch(engine)
        if torch.cuda.is_available() and 'after_train_epoch' in self._timings:
            torch.cuda.empty_cache()

    def before_val_epoch(self, engine):
        self.before_epoch(engine)
        if torch.cuda.is_available() and 'before_val_epoch' in self._timings:
            torch.cuda.empty_cache()

    def after_val_epoch(self, engine):
        self.after_epoch(engine)
        if torch.cuda.is_available() and 'after_val_epoch' in self._timings:
            torch.cuda.empty_cache()

    def before_train_iter(self, engine):
        self.before_iter(engine)
        if torch.cuda.is_available() and 'before_train_iter' in self._timings:
            torch.cuda.empty_cache()

    def after_train_iter(self, engine):
        self.after_iter(engine)
        if torch.cuda.is_available() and 'after_train_iter' in self._timings:
            torch.cuda.empty_cache()

    def before_val_iter(self, engine):
        self.before_iter(engine)
        if torch.cuda.is_available() and 'before_val_iter' in self._timings:
            torch.cuda.empty_cache()

    def after_val_iter(self, engine):
        self.after_iter(engine)
        if torch.cuda.is_available() and 'after_val_iter' in self._timings:
            torch.cuda.empty_cache()
