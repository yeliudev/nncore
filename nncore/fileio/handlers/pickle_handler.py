# Copyright (c) Ye Liu. All rights reserved.

import joblib
from six.moves import cPickle as pickle

from .base import FileHandler


class PickleHandler(FileHandler):

    def load_from_fileobj(self, file):
        return joblib.load(file)

    def load_from_path(self, filename):
        return super(PickleHandler, self).load_from_path(filename, mode='rb')

    def load_from_bytes(self, bytes):
        return pickle.loads(bytes)

    def dump_to_bytes(self, obj):
        return pickle.dumps(obj, protocol=2)

    def dump_to_fileobj(self, obj, file):
        joblib.dump(obj, file, protocol=2)

    def dump_to_path(self, obj, filename):
        super(PickleHandler, self).dump_to_path(obj, filename, mode='wb')
