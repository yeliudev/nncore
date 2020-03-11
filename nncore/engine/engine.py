# Copyright (c) Ye Liu. All rights reserved.

import nncore


class Engine(object):

    def __init__(self, model, logger=None, log_level='INFO', work_dir=None):
        self.model = model

        if logger is None:
            self.logger = nncore.get_logger()
            self.logger.setLevel(log_level)
        else:
            self.logger = logger

        self.work_dir = work_dir

    def _call_hook(self, fn_name):
        for hook in self._hooks:
            getattr(hook, fn_name)(self)
