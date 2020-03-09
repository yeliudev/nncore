# Copyright (c) Ye Liu. All rights reserved.

from abc import abstractmethod


class FileHandler(object):

    @abstractmethod
    def load_from_fileobj(self, file):
        pass

    @abstractmethod
    def dump_to_fileobj(self, obj, file):
        pass

    @abstractmethod
    def load_from_bytes(self, bytes):
        pass

    @abstractmethod
    def dump_to_bytes(self, obj):
        pass

    def load_from_path(self, filename, mode='r'):
        with open(filename, mode) as f:
            return self.load_from_fileobj(f)

    def dump_to_path(self, obj, filename, mode='w'):
        with open(filename, mode) as f:
            self.dump_to_fileobj(obj, f)
