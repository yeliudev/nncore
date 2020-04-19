# Copyright (c) Ye Liu. All rights reserved.

import nncore
from nncore.engine import comm
from .base import HOOKS, Hook


@HOOKS.register
class EvalHook(Hook):

    def __init__(self, interval=1, buffer_keys=[], **kwargs):
        super(EvalHook, self).__init__()
        self._interval = interval
        self._buffer_keys = buffer_keys
        self._kwargs = kwargs

    def after_val_epoch(self, engine):
        results = {
            k: v
            for k, v in engine.buffer.items() if k in self._buffer_keys
        }

        output = engine.data_loader.dataset.evaluate(
            results, logger=engine.logger, **self._kwargs)

        for key in output:
            engine.buffer.update(key, output[key])

        for key in self._buffer_keys:
            engine.buffer.remove(key)


@HOOKS.register
class DistEvalHook(EvalHook):

    def after_val_epoch(self, engine):
        results = {
            k: v
            for k, v in engine.buffer.items() if k in self._buffer_keys
        }

        results = comm.gather(results)

        if comm.is_main_process():
            results = nncore.to_dict_of_list(results)
            results = {k: nncore.concat_list(v) for k, v in results.items()}
            output = engine.data_loader.dataset.evaluate(
                results, logger=engine.logger, **self._kwargs)
        comm.synchronize()

        for key in output:
            engine.buffer.update(key, output[key])

        for key in self._buffer_keys:
            engine.buffer.remove(key)
