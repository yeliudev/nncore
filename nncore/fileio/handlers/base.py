# Copyright (c) Ye Liu. All rights reserved.

from abc import ABCMeta, abstractmethod


class FileHandler(metaclass=ABCMeta):

    @abstractmethod
    def load_from_fileobj(self):
        pass

    @abstractmethod
    def dump_to_fileobj(self):
        pass

    @abstractmethod
    def load_from_str(self):
        pass

    @abstractmethod
    def dump_to_str(self):
        pass

    def load_from_path(self, filename, mode='r', **kwargs):
        with open(filename, mode) as f:
            return self.load_from_fileobj(f, **kwargs)

    def dump_to_path(self, obj, filename, mode='w', **kwargs):
        with open(filename, mode) as f:
            self.dump_to_fileobj(obj, f, **kwargs)
