# Copyright (c) Ye Liu. Licensed under the MIT License.

import os

from setuptools import find_packages, setup

INSTALL_REQUIRES = [
    'h5py>=3.10', 'joblib>=1.3', 'jsonlines>=4', 'numpy>=1.26',
    'opencv-python>=4.9', 'pyyaml>=6', 'requests>=2.31', 'tabulate>=0.9',
    'tensorboard>=2.16', 'termcolor>=2.4'
]


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
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Topic :: Utilities',
    ],
    python_requires='>=3.9',
    install_requires=INSTALL_REQUIRES,
    extras_require={'docs': ['sphinx-rtd-theme']},
    packages=find_packages(exclude=('.github', 'docs', 'examples', 'tests')))
