# Copyright (c) Ye Liu. All rights reserved.

import nncore

HOOKS = nncore.Registry('hook')


def build_hook(cfg, **kwargs):
    """
    Build a hook from a dict.

    Args:
        cfg (dict or str): The config or name of the hook.

    Returns:
        :obj:`Hook`: The constructed hook.
    """
    return nncore.build_object(cfg, HOOKS, **kwargs)
