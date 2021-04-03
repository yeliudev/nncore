# Copyright (c) Ye Liu. All rights reserved.

from .base import Hook
from .builder import HOOKS


@HOOKS.register()
class SamplerSeedHook(Hook):
    """
    Update sampler seeds every epoch. This hook is normally used in
    distributed training.
    """

    def before_epoch(self, engine):
        engine.data_loader.sampler.set_epoch(engine.epoch)
