# Copyright (c) Ye Liu. All rights reserved.

from collections import OrderedDict

import nncore
from .hooks import Hook


@nncore.bind_getters('hooks', 'epoch', 'iter', 'inner_iter', 'max_epochs',
                     'max_iters')
class Engine(object):

    def __init__(self, model, logger=None, log_level='INFO', work_dir=None):
        self.model = model
        self.work_dir = work_dir

        if logger is None:
            self.logger = nncore.get_logger()
            self.logger.setLevel(log_level)
        else:
            self.logger = logger

        self.flush_states()

    def __call_hook(self, fn_name):
        for hook in self._hooks.values():
            getattr(hook, fn_name)(self)

    def flush_states(self):
        self._hooks = OrderedDict()
        self._epoch = 0
        self._iter = 0
        self._inner_iter = 0
        self._max_epochs = 0
        self._max_iters = 0

    def register_hook(self, hook, name=None, after=None):
        """
        Register a hook into the engine.

        Args:
            hook (:obj:`Hook`): the hook to be registered
            name (str, optional): name of the hook to be registered
            after (str, optional): name of the hook to insert after. The
                registered hook will be inserted into the end of the hook list
                by default.
        """
        assert isinstance(hook, Hook)
        name = name or getattr(hook, name, None) or hook.__class__.__name__

        if name in self._hooks:
            raise KeyError("hook '{}' has been registered before".format(name))

        self._hooks[name] = hook
        if after is not None:
            if after not in self._hooks:
                raise KeyError(
                    "hook '{}' not found in registered hooks".format(after))

            matched = False
            for key in list(self._hooks.keys()):
                if key == after:
                    matched = True
                elif matched and key != name:
                    self._hooks.move_to_end(key)
