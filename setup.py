# Copyright (c) Ye Liu. All rights reserved.

import os.path as osp

from setuptools import find_packages, setup


def get_version():
    version_file = osp.join('nncore', '__init__.py')
    with open(version_file, encoding='utf-8') as f:
        lines = f.readlines()
    version_line = [l.strip() for l in lines if l.startswith('__version__')][0]
    version = version_line.split('=')[-1].strip().strip('"\'')
    return version


def get_readme():
    with open('README.md', encoding='utf-8') as f:
        content = f.read()
    return content


setup(
    name='nncore',
    version=get_version(),
    author='Ye Liu',
    author_email='yeliudev@gmail.com',
    license='MIT',
    url='https://github.com/c1aris/nncore',
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
    install_requires=[
        'addict', 'joblib', 'pynvml', 'pyyaml', 'sentry-sdk', 'six',
        'tabulate', 'termcolor'
    ],
    packages=find_packages(exclude=('examples', 'tests')))
