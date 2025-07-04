# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
import os
import sys

sys.path.insert(0, os.path.abspath('..'))

# -- Project information -----------------------------------------------------

project = 'NNCore'
copyright = '2025, Ye Liu'
author = 'Ye Liu'


def get_version():
    version_file = os.path.join('..', 'nncore', '__init__.py')
    with open(version_file, encoding='utf-8') as f:
        lines = f.readlines()
    for line in lines:
        if line.startswith('__version__'):
            exec(line.strip())
    return locals()['__version__']


# The full version, including alpha/beta/rc tags
release = get_version()

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom ones.
extensions = [
    'sphinx.ext.autodoc', 'sphinx.ext.napoleon', 'sphinx.ext.viewcode'
]

# Mock modules to prevent unnecessary installation
autodoc_mock_imports = ['torch', 'torchvision']

# Sort the automatically documented members by source order
autodoc_member_order = 'bysource'

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files. This pattern also
# affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages. See the documentation for a
# list of builtin themes.
html_theme = 'sphinx_rtd_theme'
