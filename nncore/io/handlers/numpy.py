# Copyright (c) Ye Liu. All rights reserved.

import numpy as np

from .base import FileHandler


class NPYHandler(FileHandler):
    """
    Handler for Numpy files.
    """

    def load_from_path(self, file, **kwargs):
        return np.load(file, **kwargs)

    def dump_to_path(self, obj, file, **kwargs):
        np.save(file, obj, **kwargs)
