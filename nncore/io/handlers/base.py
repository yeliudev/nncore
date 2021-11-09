# Copyright (c) Ye Liu. All rights reserved.


class FileHandler(object):
    """
    Base class for file handlers. The inherited classes can optionally override
    :obj:`load_from_file`, :obj:`dump_to_file`, :obj:`load_from_str`, and
    :obj:`dump_to_str` methods to support loading or dumping data.
    """

    def load_from_file(self):
        raise NotImplementedError

    def dump_to_file(self):
        raise NotImplementedError

    def load_from_str(self):
        raise NotImplementedError

    def dump_to_str(self):
        raise NotImplementedError

    def load_from_path(self, file, mode='r', **kwargs):
        with open(file, mode) as f:
            return self.load_from_file(f, **kwargs)

    def dump_to_path(self, obj, file, mode='w', **kwargs):
        with open(file, mode) as f:
            self.dump_to_file(obj, f, **kwargs)
