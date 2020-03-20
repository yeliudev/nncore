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
        self._interval = interval
        self._save_optimizer = save_optimizer
        self._create_symlink = create_symlink
        self._out_dir = out_dir

    def on_register(self, engine):
        if self._out_dir is None:
            self._out_dir = engine.work_dir

        if not nncore.dir_exist(self._out_dir):
            raise ValueError("invalid out_dir: {}".format(self._out_dir))

    @master_only
    def after_train_epoch(self, engine):
        if not self.every_n_epochs(engine, self._interval):
            return

        filename = 'epoch_{}.pth'.format(engine.epoch + 1)
        filepath = osp.join(self._out_dir, filename)
        optimizer = engine.optimizer if self._save_optimizer else None
        meta = dict(epoch=engine.epoch + 1, iter=engine.iter)

        engine.logger.info('Saving checkpoint to {}...'.format(filepath))
        save_checkpoint(engine.model, filepath, optimizer=optimizer, meta=meta)

        if self._create_symlink:
            nncore.symlink(filename, osp.join(self._out_dir, 'latest.pth'))
