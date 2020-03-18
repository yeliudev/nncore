# Copyright (c) Ye Liu. All rights reserved.

import os.path as osp

import nncore
from ..comm import master_only
from ..utils import save_checkpoint
from .base import HOOKS, Hook


@HOOKS.register
class CheckpointHook(Hook):

    def __init__(self,
                 interval=1,
                 save_optimizer=True,
                 create_symlink=True,
                 out_dir=None):
        super(CheckpointHook, self).__init__()
        self.interval = interval
        self.save_optimizer = save_optimizer
        self.create_symlink = create_symlink
        self.out_dir = out_dir

    def _save_checkpoint(self, engine, meta):
        filename = 'epoch_{}.pth'.format(engine.epoch + 1)
        filepath = osp.join(self.out_dir, filename)
        optimizer = engine.optimizer if self.save_optimizer else None

        engine.logger.info('Saving checkpoint to {}...'.format(filepath))
        save_checkpoint(engine.model, filepath, optimizer=optimizer, meta=meta)

        if self.create_symlink:
            nncore.symlink(filename, osp.join(self.out_dir, 'latest.pth'))

    def on_register(self, engine):
        if self.out_dir is None:
            if not nncore.dir_exist(engine.work_dir):
                raise ValueError("invalid work_dir: {}".format(
                    engine.work_dir))
            self.out_dir = engine.work_dir

    @master_only
    def after_train_epoch(self, engine):
        if not self.every_n_epochs(engine, self.interval):
            return

        meta = dict(epoch=engine.epoch + 1, iter=engine.iter)
        self._save_checkpoint(engine, meta)
