# Copyright (c) Ye Liu. All rights reserved.

from .base import HOOKS, Hook


@HOOKS.register()
class SamplerSeedHook(Hook):

    def before_epoch(self, engine):
        engine.data_loader.sampler.set_epoch(engine.epoch)
