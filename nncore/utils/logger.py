# Copyright (c) Ye Liu. All rights reserved.

import logging
import sys

from termcolor import colored

_CACHED_LOGGER = []

_COLOR_MAP = {
    0: 'white',
    10: 'cyan',
    20: 'green',
    30: 'yellow',
    40: 'red',
    50: 'red'
}


class _Formatter(logging.Formatter):

    def formatMessage(self, record):
        log = super(_Formatter, self).formatMessage(record)
        anchor = -len(record.message)
        prefix = colored(log[:anchor], color=_COLOR_MAP[record.levelno])
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
        fmt (str, optional): log format. The format must end with `%(message)s`
            to make sure that the colors could be rendered properly.
        datefmt (str, optional): date format
        log_level (str or int, optional): log level of the logger. Note that
            only the main process (rank 0) is affected, and other processes
            will set the level to `ERROR` thus be silent at most of the time.
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
        from nncore.engine import is_main_process
        if not is_main_process():
            logger.setLevel(logging.ERROR)
            return logger
    except ImportError:
        pass

    if log_file is not None:
        fh = logging.FileHandler(log_file, mode='w')
        formatter = logging.Formatter(fmt=fmt, datefmt=datefmt)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


def log_or_print(msg, logger, log_level=logging.INFO):
    """
    Log a message with a potential logger. If `logger` is a valid
    `logging.Logger` or a name of the logger, then it would be used. Otherwise
    this method will use the normal `print` function instead.

    Args:
        msg (str): the message to be logged
        logger (any): the potential logger or name of the logger to use
        log_level (int, optional): log level of the logger
    """
    level = logging._checkLevel(log_level)
    if isinstance(logger, logging.Logger):
        logger.log(level, msg)
    elif isinstance(logger, str):
        logger = logging.getLogger(logger)
        logger.log(level, msg)
    else:
        if level > 20:
            level_name = logging._levelToName[level]
            msg = '{} {}'.format(
                colored(level_name + ':', color=_COLOR_MAP[level]), msg)
        print(msg)
