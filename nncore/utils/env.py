# Copyright (c) Ye Liu. Licensed under the MIT License.

import importlib
import os
import platform
import re
import subprocess
import sys
import time
from collections import defaultdict
from getpass import getuser
from socket import gethostname

from tabulate import tabulate

from .path import dir_name, is_dir, join


def get_host_info():
    return '{}@{}'.format(getuser(), gethostname())


def get_time_str():
    return time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())


def get_timestamp():
    return time.strftime('%Y%m%d%H%M%S', time.localtime())


def exec(cmd):
    return subprocess.check_output(cmd, shell=True).decode().strip()


def _collect_cpu_env(system_type):
    if system_type == 'Linux':
        info = exec('cat /proc/cpuinfo')
        for line in info.split('\n'):
            if 'model name' in line:
                return re.sub('.*model name.*:', '', line, 1)
    elif system_type == 'Darwin':
        return exec('sysctl -n machdep.cpu.brand_string')
    elif system_type == 'Windows':
        return platform.processor()
    return '<unknown>'


def _collect_cuda_env():
    try:
        import torch
        from torch.utils.cpp_extension import CUDA_HOME
        if CUDA_HOME is not None and is_dir(CUDA_HOME):
            try:
                nvcc = join(CUDA_HOME, 'bin', 'nvcc')
                nvcc = exec("'{}' -V | tail -n1".format(nvcc))
            except subprocess.SubprocessError:
                nvcc = None
        else:
            nvcc = None
        if torch.cuda.is_available():
            devices = defaultdict(list)
            for k in range(torch.cuda.device_count()):
                devices[torch.cuda.get_device_name(k)].append(str(k))
        else:
            devices = None
        return CUDA_HOME, nvcc, devices
    except ImportError:
        return None, None, None


def _collect_torch_env():
    try:
        import torch
        version = torch.__version__
        root = dir_name(torch.__file__)
        return '{} @ {}'.format(version, root), torch.version.debug
    except ImportError:
        return None, None


def _collect_torch_build_env():
    try:
        try:
            import torch
            return torch.__config__.show()
        except ImportError:
            from torch.utils.collect_env import get_pretty_env_info
            return get_pretty_env_info()
    except ImportError:
        pass


def _detect_compute_compatibility(cuda_home, so_file):
    try:
        cuobjdump = os.path.join(cuda_home, 'bin', 'cuobjdump')
        if os.path.isfile(cuobjdump):
            output = exec("'{}' --list-elf '{}'".format(cuobjdump, so_file))
            sm = []
            for line in output.split('\n'):
                line = re.findall(r'\.sm_[0-9]*\.', line)[0]
                sm.append(line.strip('.'))
            sm = sorted(set(sm))
            return ', '.join(sm)
        else:
            return so_file + '; cannot find cuobjdump'
    except Exception:
        return so_file


def _collect_torchvision_env():
    try:
        import torch
        import torchvision
        from torch.utils.cpp_extension import CUDA_HOME
        torchvision_env = '{} @ {}'.format(torchvision.__version__,
                                           dir_name(torchvision.__file__))
        if torch.cuda.is_available():
            try:
                torchvision_C = importlib.util.find_spec(
                    'torchvision._C').origin
                torchvision_arch_flags = _detect_compute_compatibility(
                    CUDA_HOME, torchvision_C)
                return torchvision_env, torchvision_arch_flags
            except ImportError:
                pass
        return torchvision_env, None
    except ImportError:
        return None, None


def _get_module_version(mod_name):
    try:
        mod = importlib.import_module(mod_name)
        return mod.__version__
    except ImportError:
        pass


def collect_env_info(modules=['numpy', 'PIL', 'cv2']):
    """
    Collect information about the environment.

    This method will try and collect all the information about the entire
    environment, including platform, Python version, CUDA version, PyTorch
    version, etc., and return a str describing the environment.

    Args:
        modules (list[str], optional): The list of module names to be checked.
            Default: ``['numpy', 'PIL', 'cv2']``.

    Returns:
        str: The environment information.
    """
    info = []

    system_type = platform.system()
    info.append(('System', system_type))
    info.append(('Python', sys.version.replace('\n', '')))
    info.append(('CPU', _collect_cpu_env(system_type)))

    cuda_home, nvcc, devices = _collect_cuda_env()
    if cuda_home is not None:
        info.append(('CUDA_HOME', cuda_home))
        info.append(('NVCC', nvcc or '<not-found>'))
        if devices is not None:
            for name, ids in devices.items():
                info.append(('GPU ' + ','.join(ids), name))
        else:
            info.append(('GPU', '<not-found>'))
    else:
        info.append(('CUDA', '<not-found>'))

    torch_env, torch_debug_build = _collect_torch_env()
    info.append(('PyTorch', torch_env or '<not-found>'))
    if torch_debug_build is not None:
        info.append(('PyTorch debug build', torch_debug_build))

    torchvision_env, torchvision_arch_flags = _collect_torchvision_env()
    info.append(('torchvision', torchvision_env or '<not-found>'))
    if torchvision_arch_flags is not None:
        info.append(('torchvision arch flags', torchvision_arch_flags))

    for module in ['nncore'] + modules:
        info.append((module, _get_module_version(module) or '<not-found>'))

    env_info = tabulate(info)
    torch_build_env = _collect_torch_build_env()
    if torch_build_env is not None:
        env_info += '\n{}'.format(torch_build_env)

    return env_info
