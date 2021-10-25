# Copyright (c) Ye Liu. All rights reserved.

from .base import FileHandler


class TXTHandler(FileHandler):
    """
    Handler for plain text files.
    """

    def load_from_file(self, file, offset=0, separator=None, max_length=-1):
        out, count = [], 0

        for _ in range(offset):
            file.readline()

        for line in file:
            if max_length >= 0 and count >= max_length:
                break

            line = line.rstrip('\n')
            if separator is not None:
                line = line.split(separator)

            out.append(line)
            count += 1

        return out

    def dump_to_file(self, obj, file, separator=','):
        if isinstance(obj, (list, tuple)):
            tmp = [
                separator.join(o) if isinstance(o, (list, tuple)) else o
                for o in obj
            ]
            file.write('\n'.join(tmp))
        else:
            file.write(str(obj))
