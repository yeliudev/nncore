# Copyright (c) Ye Liu. All rights reserved.

import logging
import sys

from termcolor import colored

_CACHED_LOGGER = []


class _Formatter(logging.Formatter):

    _color_map = {
        0: 'white',
        10: 'cyan',
        20: 'green',
        30: 'yellow',
        40: 'red',
        50: 'red'
    }

    def formatMessage(self, record):
        log = super(_Formatter, self).formatMessage(record)
        anchor = -len(record.message)
        prefix = colored(log[:anchor], color=self._color_map[record.levelno])
        return prefix + log[anchor:]


def get_logger(name='nncore',
               fmt='[%(asctime)s %(levelname)s]: %(message)s',
               datefmt='%Y-%m-%d %H:%M:%S',
               log_level=logging.INFO,
               log_file=None):
    """
    Initializes and get a logger by name.

    If the logger has not been initialized, this method will initialize the
    logger by adding one or two handlers, otherwise the initialized logger will
    be directly returned. During initialization, a StreamHandler will always be
    added. If `log_file` is specified and the process rank is 0, a FileHandler
    will also be added.

    Args:
        name (str, optional): logger name
        fmt (str, optional): log format. The format must end with '%(message)s'
            to make sure that the colors could be rendered properly.
        datefmt (str, optional): date format
        log_level (int, optional): the logger level. Note that only the main
            process (rank 0) is affected, and other processes will set the
            level to 'ERROR' thus be silent most of the time.
        log_file (str, optional): filename of the log file. If not None, a
            FileHandler will be added to the logger.

    Returns:
        logger (logging.Logger): the expected logger
    """
    logger = logging.getLogger(name)

    if name in _CACHED_LOGGER:
        return logger

    _CACHED_LOGGER.append(name)

    logger.setLevel(log_level)
    logger.propagate = False

    sh = logging.StreamHandler(stream=sys.stdout)
    formatter = _Formatter(fmt=fmt, datefmt=datefmt)
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    try:
        from nncore.engine import comm
        if not comm.is_main_process():
            logger.setLevel(logging.ERROR)
            return logger
    except ModuleNotFoundError:
        pass

    if log_file is not None:
        fh = logging.FileHandler(log_file, mode='w')
        formatter = logging.Formatter(fmt=fmt, datefmt=datefmt)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger
