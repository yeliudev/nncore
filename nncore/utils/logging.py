# Copyright (c) Ye Liu. Licensed under the MIT License.

import logging
import sys

from termcolor import colored

from .path import abs_path, dir_name, mkdir

_COLOR_MAP = {
    0: 'white',
    10: 'cyan',
    20: 'green',
    30: 'yellow',
    40: 'red',
    50: 'magenta'
}

_CACHED_LOGGERS = []
_DEFAULT_LOGGER = None


class _Formatter(logging.Formatter):

    def formatMessage(self, record):
        log = super(_Formatter, self).formatMessage(record)
        anchor = -len(record.message)
        prefix = colored(log[:anchor], color=_COLOR_MAP[record.levelno])
        return prefix + log[anchor:]


def get_logger(logger=None,
               fmt='[%(asctime)s %(levelname)s]: %(message)s',
               datefmt='%Y-%m-%d %H:%M:%S',
               log_level=logging.INFO,
               log_file=None):
    """
    Initialize and get a logger by name.

    If the logger has not been initialized, this method will initialize the
    logger by adding one or two handlers, otherwise the initialized logger will
    be directly returned. During initialization, a :obj:`StreamHandler` will
    always be added. If ``log_file`` is specified and the process rank is
    ``0``, a :obj:`FileHandler` will also be added.

    Args:
        logger (:obj:`logging.Logger` | str | None, optional): The logger or
            name of the logger. Default: ``None``.
        fmt (str, optional): Logging format of the logger. The format must end
            with ``'%(message)s'`` to make sure that the colors can be rendered
            properly. Default: ``'[%(asctime)s %(levelname)s]: %(message)s'``.
        datefmt (str, optional): Date format of the logger. Default:
            ``'%Y-%m-%d %H:%M:%S'``.
        log_level (str | int, optional): Log level of the logger. Note that
            only the main process (rank 0) is affected, and other processes
            will set the level to ``ERROR`` thus be silent at most of the time.
            Default: :obj:`logging.INFO`.
        log_file (str | None, optional): Path to the log file. If specified, a
            :obj:`FileHandler` will be added to the logger of the main process.
            Default: ``None``.

    Returns:
        :obj:`logging.Logger`: The expected logger.
    """
    global _DEFAULT_LOGGER

    if isinstance(logger, logging.Logger):
        return logger

    logger = logging.getLogger(logger)

    if logger in _CACHED_LOGGERS:
        return logger

    _CACHED_LOGGERS.append(logger)

    if _DEFAULT_LOGGER is None:
        _DEFAULT_LOGGER = logger

    logger.setLevel(log_level)
    logger.propagate = False

    if fmt is not None and not fmt.endswith('%(message)s'):
        raise ValueError("fmt must end with '%(message)s'")

    formatter = _Formatter(fmt=fmt, datefmt=datefmt)
    sh = logging.StreamHandler(stream=sys.stdout)
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
        mkdir(dir_name(abs_path(log_file)))
        fh = logging.FileHandler(log_file, mode='w')
        formatter = logging.Formatter(fmt=fmt, datefmt=datefmt)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


def set_default_logger(logger=None,
                       fmt='[%(asctime)s %(levelname)s]: %(message)s',
                       datefmt='%Y-%m-%d %H:%M:%S',
                       log_level=logging.INFO,
                       log_file=None):
    """
    Initialize and set the default logger.

    If the logger has not been initialized, this method will initialize the
    logger by adding one or two handlers, otherwise the initialized logger will
    be directly set. During initialization, a :obj:`StreamHandler` will
    always be added. If ``log_file`` is specified and the process rank is
    ``0``, a :obj:`FileHandler` will also be added.

    Args:
        logger (:obj:`logging.Logger` | str | None, optional): The logger or
            name of the logger. Default: ``None``.
        fmt (str, optional): Logging format of the logger. The format must end
            with ``'%(message)s'`` to make sure that the colors can be rendered
            properly. Default: ``'[%(asctime)s %(levelname)s]: %(message)s'``.
        datefmt (str, optional): Date format of the logger. Default:
            ``'%Y-%m-%d %H:%M:%S'``.
        log_level (str | int, optional): Log level of the logger. Note that
            only the main process (rank 0) is affected, and other processes
            will set the level to ``ERROR`` thus be silent at most of the time.
            Default: :obj:`logging.INFO`.
        log_file (str | None, optional): Path to the log file. If specified, a
            :obj:`FileHandler` will be added to the logger of the main process.
            Default: ``None``.
    """
    global _DEFAULT_LOGGER

    _DEFAULT_LOGGER = get_logger(
        logger=logger,
        fmt=fmt,
        datefmt=datefmt,
        log_level=log_level,
        log_file=log_file)


def log(*args, logger=None, log_level=logging.INFO, **kwargs):
    """
    Print a message with a logger. If ``logger`` is a valid
    :obj:`logging.Logger` or a name of the logger, then it would be used.
    Otherwise this method will try to use the default logger or the normal
    :obj:`print` function instead.

    Args:
        *args (list[str] | str): The message to be logged.
        logger (:obj:`logging.Logger` | str | None, optional): The logger
            or name of the logger to use. Default: ``None``.
        log_level (int, optional): Log level of the logger. Default:
            :obj:`logging.INFO`.
    """
    level = logging._checkLevel(log_level)
    msg = ' '.join([str(a) for a in args])
    if isinstance(logger, logging.Logger):
        logger.log(level, msg)
    elif isinstance(logger, str):
        logger = get_logger(logger, **kwargs)
        logger.log(level, msg)
    elif isinstance(_DEFAULT_LOGGER, logging.Logger):
        _DEFAULT_LOGGER.log(level, msg)
    else:
        if level > 20:
            level_name = logging._levelToName[level]
            msg = '{} {}'.format(
                colored(level_name + ':', color=_COLOR_MAP[level]), msg)
        print(msg)
