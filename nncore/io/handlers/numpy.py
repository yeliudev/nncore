# Copyright (c) Ye Liu. Licensed under the MIT License.

import numpy as np

from .base import FileHandler


class NumPyHandler(FileHandler):
    """
    Handler for NumPy files.
    """

    def load_from_file(self, file, **kwargs):
        return np.load(file, **kwargs)

    def dump_to_file(self, obj, file, **kwargs):
        np.save(file, obj, **kwargs)

    def load_from_path(self, path, **kwargs):
        return self.load_from_file(path, **kwargs)

    def dump_to_path(self, obj, path, **kwargs):
        self.dump_to_file(obj, path, **kwargs)


class NumPyzHandler(NumPyHandler):
    """
    Handler for compressed NumPy files.
    """

    def dump_to_file(self, obj, file, **kwargs):
        np.savez(file, obj, **kwargs)
