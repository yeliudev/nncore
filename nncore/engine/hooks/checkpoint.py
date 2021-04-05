# Copyright (c) Ye Liu. All rights reserved.

import nncore
from ..comm import master_only
from ..utils import save_checkpoint
from .base import Hook
from .builder import HOOKS


@HOOKS.register()
class CheckpointHook(Hook):
    """
    Save checkpoints periodically during training. Checkpoint of the last
    epoch will always be saved regardless of ``interval``.

    Args:
        interval (int, optional): The interval of epochs to save checkpoints.
            Default: ``1``.
        save_optimizer (bool, optional): Whether to incorperate optimizer
            statuses into checkpoints. Default: ``True``.
        create_symlink (bool, optional): Whether to create a symlink to the
            latest checkpoint. This argument is invalid on Windows due to the
            limitations of its file system. Default: ``False``.
        out (str or None, optional): Path to the output directory. If not
            specified, :obj:`enging.work_dir` will be used as the default path.
            Default: ``None``.
    """

    def __init__(self,
                 interval=1,
                 save_optimizer=True,
                 create_symlink=False,
                 out=None):
        super(CheckpointHook, self).__init__()
        self._interval = interval
        self._save_optimizer = save_optimizer
        self._create_symlink = create_symlink
        self._out = out

    @master_only
    def before_launch(self, engine):
        if self._out is None:
            self._out = engine.work_dir
        nncore.mkdir(self._out)

    @master_only
    def after_train_epoch(self, engine):
        if not self.last_epoch(engine) and not self.every_n_epochs(
                engine, self._interval):
            return

        filename = 'epoch_{}.pth'.format(engine.epoch + 1)
        filepath = nncore.join(self._out, filename)
        optimizer = engine.optimizer if self._save_optimizer else None

        meta = dict(
            epoch=engine.epoch + 1,
            iter=engine.iter,
            stages=[
                stage.to_dict() if isinstance(stage, nncore.CfgNode) else stage
                for stage in engine.stages
            ])

        engine.logger.info('Saving checkpoint to {}...'.format(filepath))
        save_checkpoint(engine.model, filepath, optimizer=optimizer, meta=meta)

        if self._create_symlink:
            nncore.symlink(filename, nncore.join(self._out, 'latest.pth'))
