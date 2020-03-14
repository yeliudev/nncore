# Copyright (c) Ye Liu. All rights reserved.

import importlib
import os
import os.path as osp
import subprocess
import sys
from getpass import getuser
from platform import system
from re import findall
from socket import gethostname

from tabulate import tabulate


def get_host_info():
    return '{}@{}'.format(getuser(), gethostname())


def _detect_compute_compatibility(CUDA_HOME, so_file):
    try:
        cuobjdump = os.path.join(CUDA_HOME, 'bin', 'cuobjdump')
        if os.path.isfile(cuobjdump):
            output = subprocess.check_output(
                "'{}' --list-elf '{}'".format(cuobjdump, so_file), shell=True)
            output = output.decode('utf-8').strip().split('\n')
            sm = []
            for line in output:
                line = findall(r'\.sm_[0-9]*\.', line)[0]
                sm.append(line.strip('.'))
            sm = sorted(set(sm))
            return ', '.join(sm)
        else:
            return so_file + '; cannot find cuobjdump'
    except Exception:
        return so_file


def _collect_cuda_env():
    try:
        import torch
        from torch.utils.cpp_extension import CUDA_HOME
        cuda_arch_list = os.environ.get('TORCH_CUDA_ARCH_LIST', None)
        if CUDA_HOME is not None and osp.isdir(CUDA_HOME):
            try:
                nvcc = osp.join(CUDA_HOME, 'bin', 'nvcc')
                nvcc = subprocess.check_output(
                    "'{}' -V | tail -n1".format(nvcc), shell=True)
                nvcc = nvcc.decode('utf-8').strip()
            except subprocess.SubprocessError:
                nvcc = None
        else:
            nvcc = None
        if torch.cuda.is_available():
            devices = {}
            for k in range(torch.cuda.device_count()):
                devices[torch.cuda.get_device_name(k)].append(str(k))
        else:
            devices = None
        return CUDA_HOME, cuda_arch_list, nvcc, devices
    except ImportError:
        return None, None, None, None


def _collect_torch_env():
    try:
        import torch
        return '{} @ {}'.format(torch.__version__, osp.dirname(
            torch.__file__)), torch.version.debug
    except ImportError:
        return None, None


def _collect_torch_build_env():
    try:
        try:
            import torch.__config__
            return torch.__config__.show()
        except ImportError:
            from torch.utils.collect_env import get_pretty_env_info
            return get_pretty_env_info()
    except ImportError:
        return None


def _collect_torchvision_env():
    try:
        import torch
        import torchvision
        from torch.utils.cpp_extension import CUDA_HOME
        torchvision_env = '{} @ {}'.format(torchvision.__version__,
                                           osp.dirname(torchvision.__file__))
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


def _get_module_version(module_name):
    try:
        module = importlib.import_module(module_name)
        return module.__version__
    except ImportError:
        return None


def collect_env_info():
    """
    Collect information about the environment.

    This method will try and collect all the information about the entire
    environment, including platform, python version, cuda version, pytorch
    version, etc., and return a string describing the environment.

    Returns:
        info (str): the information about the environment
    """
    _c = []

    # system info
    _c.append(('System', system()))
    _c.append(('Python', sys.version.replace('\n', '')))

    # cuda info
    cuda_home, cuda_arch_list, nvcc, devices = _collect_cuda_env()
    if cuda_home is not None:
        _c.append(('CUDA_HOME', cuda_home))
        _c.append(('TORCH_CUDA_ARCH_LIST', cuda_arch_list or '<not found>'))
        _c.append(('NVCC', nvcc or '<not found>'))
        if devices is not None:
            for name, ids in devices.items():
                _c.append(('GPU ' + ','.join(ids), name))
        else:
            _c.append(('GPU', '<not found>'))
    else:
        _c.append(('CUDA', '<not found>'))

    # pytorch info
    torch_env, torch_debug_build = _collect_torch_env()
    _c.append(('PyTorch', torch_env or '<not found>'))
    if torch_debug_build is not None:
        _c.append(('PyTorch debug build', torch_debug_build))

    # torchvison info
    torchvision_env, torchvision_arch_flags = _collect_torchvision_env()
    _c.append(('torchvision', torchvision_env or '<not found>'))
    if torchvision_arch_flags is not None:
        _c.append(('torchvision arch flags', torchvision_arch_flags))

    # other libraries
    for module in ['nncore', 'numpy', 'PIL', 'cv2']:
        _c.append((module, _get_module_version(module)))

    env_info = tabulate(_c)
    torch_build_env = _collect_torch_build_env()
    if torch_build_env is not None:
        env_info += '\n{}'.format(torch_build_env)

    return env_info
