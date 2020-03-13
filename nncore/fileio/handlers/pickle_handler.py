# Copyright (c) Ye Liu. All rights reserved.

import joblib
from six.moves import cPickle as pickle

from .base import FileHandler


class PickleHandler(FileHandler):

    def load_from_fileobj(self, file, **kwargs):
        return joblib.load(file, **kwargs)

    def load_from_path(self, filename, **kwargs):
        return super(PickleHandler, self).load_from_path(
            filename, mode='rb', **kwargs)

    def load_from_bytes(self, bytes, **kwargs):
        return pickle.loads(bytes, **kwargs)

    def dump_to_bytes(self, obj, protocol=2, **kwargs):
        return pickle.dumps(obj, protocol=protocol, **kwargs)

    def dump_to_fileobj(self, obj, file, protocol=2, **kwargs):
        joblib.dump(obj, file, protocol=protocol, **kwargs)

    def dump_to_path(self, obj, filename, **kwargs):
        super(PickleHandler, self).dump_to_path(
            obj, filename, mode='wb', **kwargs)
