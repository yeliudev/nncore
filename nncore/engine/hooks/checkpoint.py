# Copyright (c) Ye Liu. All rights reserved.

import os.path as osp

import nncore
from ..comm import master_only
from ..utils import save_checkpoint
from .base import HOOKS, Hook
from .utils import every_n_epochs, every_n_steps


@HOOKS.register
class CheckpointHook(Hook):

    def __init__(self,
                 interval=1,
                 interval_type='epoch',
                 filename_tmpl=None,
                 save_optimizer=True,
                 create_symlink=True,
                 out_dir=None):
        assert interval_type in ['epoch', 'iter']

        self.interval = interval
        self.interval_type = interval_type
        self.save_optimizer = save_optimizer
        self.create_symlink = create_symlink
        self.out_dir = out_dir

        if filename_tmpl is None:
            self.filename_tmpl = interval_type + '_{}.pth'

    def _save_checkpoint(self, engine, meta):
        filename = self.filename_tmpl.format(self.epoch + 1)
        filepath = osp.join(self.out_dir, filename)
        optimizer = engine.optimizer if self.save_optimizer else None

        engine.logger.info('Saving checkpoint to {}...'.format(filepath))
        save_checkpoint(engine.model, filepath, optimizer=optimizer, meta=meta)

        if self.create_symlink:
            nncore.symlink(filename, osp.join(self.out_dir, 'latest.pth'))

    def on_register(self, engine):
        if not self.out_dir:
            if not nncore.dir_exist(engine.work_dir):
                raise ValueError("invalid work_dir: {}".format(
                    engine.work_dir))
            self.out_dir = engine.work_dir

    @master_only
    @every_n_epochs('interval')
    def after_train_epoch(self, engine):
        if not self.interval_type == 'epoch':
            return
        meta = dict(epoch=engine.epoch + 1, iter=engine.iter)
        self._save_checkpoint(engine, meta)

    @master_only
    @every_n_steps('interval')
    def after_train_step(self, engine):
        if not self.interval_type == 'iter':
            return
        meta = dict(iter=engine.iter + 1)
        self._save_checkpoint(engine, meta)
