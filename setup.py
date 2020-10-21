# Copyright (c) Ye Liu. All rights reserved.

import os.path as osp
import re
from platform import system

from setuptools import find_packages, setup

from pkg_resources import DistributionNotFound, get_distribution

INSTALL_REQUIRES = [
    'joblib>=0.17', 'numpy>=1.19', 'pynvml>=8', 'pyyaml>=5.3', 'six>=1.15',
    'tabulate>=0.8', 'termcolor>=1.1'
]

OPENCV_INSTALL_REQUIRES = 'opencv-python-headless>=3', 'opencv-python>=3'


def get_version():
    version_file = osp.join('nncore', '__init__.py')
    with open(version_file, encoding='utf-8') as f:
        lines = f.readlines()
    for line in lines:
        if line.startswith('__version__'):
            version_line = line.strip()
    version = version_line.split('=')[-1].strip().strip('"\'')
    return version


def get_readme():
    with open('README.md', encoding='utf-8') as f:
        content = f.read()
    return content


def install_requires():

    def _choose_requirement(primary, secondary):
        try:
            name = re.split(r'[!<>=]', primary)[0]
            get_distribution(name)
        except DistributionNotFound:
            return secondary
        return primary

    install_requires = INSTALL_REQUIRES

    if system() != 'Windows':
        primary, secondary = OPENCV_INSTALL_REQUIRES
        try:
            get_distribution(re.split(r'[!<>=]', primary)[0])
            install_requires.append(primary)
        except DistributionNotFound:
            install_requires.append(secondary)

    return install_requires


setup(
    name='nncore',
    version=get_version(),
    author='Ye Liu',
    author_email='yeliudev@outlook.com',
    license='MIT',
    url='https://github.com/yeliudev/nncore',
    description='A lightweight PyTorch code wrapper for ML researchers',
    long_description=get_readme(),
    long_description_content_type='text/markdown',
    classifiers=[
        'Development Status :: 4 - Beta',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Topic :: Utilities',
    ],
    python_requires='>=3.6',
    setup_requires=['pytest-runner'],
    tests_require=['pytest'],
    install_requires=install_requires(),
    packages=find_packages(exclude=('.github', 'examples', 'tests')))
