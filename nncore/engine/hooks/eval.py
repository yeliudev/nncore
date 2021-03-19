# Copyright (c) Ye Liu. All rights reserved.

import nncore
from nncore.engine import comm
from .base import HOOKS, Hook


@HOOKS.register()
class EvalHook(Hook):

    def __init__(self, interval=1, buffer_key='val', **kwargs):
        super(EvalHook, self).__init__()
        self._buffer_key = buffer_key
        self._interval = interval
        self._kwargs = kwargs

    def after_val_epoch(self, engine):
        if not self.every_n_epochs(engine, self._interval):
            return

        if isinstance(self._buffer_key, (list, tuple)):
            blob = {k: engine.buffer.pop(k) for k in self._buffer_key}
        else:
            blob = engine.buffer.pop(self._buffer_key)

        output = engine.data_loader.dataset.evaluate(
            blob, logger=engine.logger, **self._kwargs)

        for key in output:
            engine.buffer.update(key, output[key])


@HOOKS.register()
class DistEvalHook(EvalHook):

    def after_val_epoch(self, engine):
        if not self.every_n_epochs(engine, self._interval):
            return

        if isinstance(self._buffer_key, (list, tuple)):
            blob = {k: engine.buffer.pop(k) for k in self._buffer_key}
        else:
            blob = engine.buffer.pop(self._buffer_key)

        blob = comm.gather(blob)

        if comm.is_main_process():
            blob = nncore.to_dict_of_list(blob)
            blob = {k: nncore.concat_list(v) for k, v in blob.items()}
            output = engine.data_loader.dataset.evaluate(
                blob, logger=engine.logger, **self._kwargs)
        else:
            output = dict()

        comm.synchronize()

        for key in output:
            engine.buffer.update(key, output[key])
