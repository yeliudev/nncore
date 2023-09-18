# Copyright (c) Ye Liu. Licensed under the MIT License.

import os
import re
from platform import system

from pkg_resources import DistributionNotFound, get_distribution
from setuptools import find_packages, setup

INSTALL_REQUIRES = [
    'h5py>=3.1', 'joblib>=1.1', 'jsonlines>=2', 'numpy>=1.19', 'pyyaml>=6',
    'tabulate>=0.8', 'tensorboard>=2.12', 'termcolor>=1.1'
]

OPENCV_INSTALL_REQUIRES = 'opencv-python-headless>=4.5', 'opencv-python>=4.5'


def get_version():
    version_file = os.path.join('nncore', '__init__.py')
    with open(version_file, encoding='utf-8') as f:
        lines = f.readlines()
    for line in lines:
        if line.startswith('__version__'):
            exec(line.strip())
    return locals()['__version__']


def get_readme():
    with open('README.md', encoding='utf-8') as f:
        content = f.read()
    return content


def get_install_requires():
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
    description='A lightweight machine learning toolkit for researchers.',
    long_description=get_readme(),
    long_description_content_type='text/markdown',
    classifiers=[
        'Development Status :: 4 - Beta',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Topic :: Utilities',
    ],
    python_requires='>=3.8',
    setup_requires=['pytest-runner'],
    tests_require=['pytest'],
    install_requires=get_install_requires(),
    packages=find_packages(exclude=('.github', 'docs', 'examples', 'tests')))
