# Copyright (c) Ye Liu. All rights reserved.

import h5py

from .base import FileHandler


class Hdf5Handler(FileHandler):

    def load_from_fileobj(self, file, dataset, **kwargs):
        obj = file.get(dataset, **kwargs)
        if isinstance(obj, h5py.Dataset):
            obj = obj[:]
        return obj

    def dump_to_fileobj(self, obj, file, dataset, **kwargs):
        file.create_dataset(dataset, data=obj, **kwargs)

    def load_from_str(self):
        raise NotImplementedError("hdf5 files do not support 'loads' method")

    def dump_to_str(self):
        raise NotImplementedError("hdf5 files do not support 'dumps' method")

    def load_from_path(self, filename, mode='r', **kwargs):
        with h5py.File(filename, mode=mode) as f:
            return self.load_from_fileobj(f, **kwargs)

    def dump_to_path(self, obj, filename, mode='w', **kwargs):
        with h5py.File(filename, mode=mode) as f:
            self.dump_to_fileobj(obj, f, **kwargs)
