# Copyright (c) Ye Liu. All rights reserved.

import pickle

import joblib

from .base import FileHandler


class PickleHandler(FileHandler):
    """
    Handler for Pickle files.
    """

    def load_from_file(self, file, **kwargs):
        return joblib.load(file, **kwargs)

    def dump_to_file(self, obj, file, protocol=2, **kwargs):
        joblib.dump(obj, file, protocol=protocol, **kwargs)

    def load_from_str(self, string, **kwargs):
        return pickle.loads(string, **kwargs)

    def dump_to_str(self, obj, protocol=2, **kwargs):
        return pickle.dumps(obj, protocol=protocol, **kwargs)

    def load_from_path(self, file, **kwargs):
        return super(PickleHandler, self).load_from_path(
            file, mode='rb', **kwargs)

    def dump_to_path(self, obj, file, **kwargs):
        super(PickleHandler, self).dump_to_path(obj, file, mode='wb', **kwargs)
