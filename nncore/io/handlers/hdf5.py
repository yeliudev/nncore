# Copyright (c) Ye Liu. All rights reserved.

import h5py
import numpy as np

from .base import FileHandler


class HDF5Handler(FileHandler):
    """
    Handler for HDF5 files.
    """

    def load_from_file(self, file, dataset, **kwargs):
        obj = file.get(dataset, **kwargs)
        if isinstance(obj, h5py.Dataset):
            obj = obj[:]
        return obj

    def dump_to_file(self, obj, file, dataset, append_mode=True, **kwargs):
        if not isinstance(obj, np.ndarray):
            raise TypeError("obj must be an np.ndarray for hdf5 files, "
                            "but got '{}'".format(type(obj)))

        if dataset in file:
            ori_size = file[dataset].shape[0]
            file[dataset].resize(ori_size + obj.shape[0], axis=0)
            file[dataset][ori_size:] = obj
        else:
            if append_mode:
                kwargs.setdefault('maxshape', [None] + list(obj.shape)[1:])
            file.create_dataset(dataset, data=obj, **kwargs)

    def load_from_path(self, path, mode='r', **kwargs):
        with h5py.File(path, mode=mode) as f:
            return self.load_from_file(f, **kwargs)

    def dump_to_path(self, obj, path, mode='a', **kwargs):
        with h5py.File(path, mode=mode) as f:
            self.dump_to_file(obj, f, **kwargs)
