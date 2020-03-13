# Copyright (c) Ye Liu. All rights reserved.

from abc import abstractmethod


class FileHandler(object):

    @abstractmethod
    def load_from_fileobj(self, file, **kwargs):
        pass

    @abstractmethod
    def dump_to_fileobj(self, obj, file, **kwargs):
        pass

    @abstractmethod
    def load_from_bytes(self, bytes, **kwargs):
        pass

    @abstractmethod
    def dump_to_bytes(self, obj, **kwargs):
        pass

    def load_from_path(self, filename, mode='r', **kwargs):
        with open(filename, mode) as f:
            return self.load_from_fileobj(f, **kwargs)

    def dump_to_path(self, obj, filename, mode='w', **kwargs):
        with open(filename, mode) as f:
            self.dump_to_fileobj(obj, f, **kwargs)
