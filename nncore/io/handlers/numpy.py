# Copyright (c) Ye Liu. All rights reserved.

import numpy as np

from .base import FileHandler


class NumPyHandler(FileHandler):
    """
    Handler for NumPy files.
    """

    def load_from_file(self, file, **kwargs):
        return np.load(file, **kwargs)

    def dump_to_file(self, obj, file, format='npy', **kwargs):
        if format == 'npy':
            np.save(file, obj, **kwargs)
        else:
            np.savez(file, obj, **kwargs)

    def load_from_path(self, path, **kwargs):
        return self.load_from_file(path, **kwargs)

    def dump_to_path(self, obj, path, format='npy', **kwargs):
        self.dump_to_file(obj, path, format=format, **kwargs)
