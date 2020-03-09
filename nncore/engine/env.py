# Copyright (c) Ye Liu. All rights reserved.

import importlib
import os
import re
import subprocess
import sys
from collections import defaultdict
from getpass import getuser
from socket import gethostname

import numpy as np
import PIL
import torch
import torchvision
from tabulate import tabulate
from torch.utils.cpp_extension import CUDA_HOME


def get_host_info():
    return '{}@{}'.format(getuser(), gethostname())


def collect_env_info():
    has_cuda = torch.cuda.is_available()

    data = []
    data.append(('sys.platform', sys.platform))
    data.append(('Python', sys.version.replace('\n', '')))
    data.append(('numpy', np.__version__))

    data.append(('PyTorch',
                 torch.__version__ + ' @' + os.path.dirname(torch.__file__)))
    data.append(('PyTorch debug build', torch.version.debug))

    data.append(('CUDA available', has_cuda))
    if has_cuda:
        devices = defaultdict(list)
        for k in range(torch.cuda.device_count()):
            devices[torch.cuda.get_device_name(k)].append(str(k))
        for name, devids in devices.items():
            data.append(('GPU ' + ','.join(devids), name))

        data.append(('CUDA_HOME', str(CUDA_HOME)))

        if CUDA_HOME is not None and os.path.isdir(CUDA_HOME):
            try:
                nvcc = os.path.join(CUDA_HOME, 'bin', 'nvcc')
                nvcc = subprocess.check_output(
                    "'{}' -V | tail -n1".format(nvcc), shell=True)
                nvcc = nvcc.decode('utf-8').strip()
            except subprocess.SubprocessError:
                nvcc = 'Not Available'
            data.append(('NVCC', nvcc))

        cuda_arch_list = os.environ.get('TORCH_CUDA_ARCH_LIST', None)
        if cuda_arch_list:
            data.append(('TORCH_CUDA_ARCH_LIST', cuda_arch_list))
    data.append(('Pillow', PIL.__version__))

    try:
        data.append((
            'torchvision',
            str(torchvision.__version__) + ' @' +
            os.path.dirname(torchvision.__file__),
        ))
        if has_cuda:
            try:
                torchvision_C = importlib.util.find_spec(
                    'torchvision._C').origin
                try:
                    cuobjdump = os.path.join(CUDA_HOME, 'bin', 'cuobjdump')
                    if os.path.isfile(cuobjdump):
                        output = subprocess.check_output(
                            "{}' --list-elf '{}'".format(
                                cuobjdump, torchvision_C),
                            shell=True)
                        output = output.decode('utf-8').strip().split('\n')
                        sm = []
                        for line in output:
                            line = re.findall(r'\.sm_[0-9]*\.', line)[0]
                            sm.append(line.strip('.'))
                        sm = sorted(set(sm))
                        arch_flags = ', '.join(sm)
                    else:
                        arch_flags = torchvision_C + '; cannot find cuobjdump'
                except Exception:
                    arch_flags = torchvision_C
                data.append(('torchvision arch flags', arch_flags))
            except ImportError:
                data.append(('torchvision._C', 'failed to find'))
    except AttributeError:
        data.append(('torchvision', 'unknown'))

    try:
        import cv2
        data.append(('cv2', cv2.__version__))
    except ImportError:
        pass

    return 'Environment info:\n{}\n{}'.format(
        tabulate(data), torch.__config__.show())
