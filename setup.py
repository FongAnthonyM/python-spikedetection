#!/usr/bin/env python
# -*- encoding: utf-8 -*-
""" setup.py
The setup for this package.
"""
# Package Header #
from src.spikedetection.header import *

# Header #
__author__ = __author__
__credits__ = __credits__
__maintainer__ = __maintainer__
__email__ = __email__


# Imports #
# Standard Libraries #
import io
import re
from glob import glob
from os.path import basename
from os.path import dirname
from os.path import join
from os.path import splitext

# Third-Party Packages #
from setuptools import find_packages
from setuptools import setup


# Definitions #
# Functions #
def read(*names, **kwargs):
    with io.open(join(dirname(__file__), *names), encoding=kwargs.get("encoding", "utf8")) as fh:
        return fh.read()


# Main #
setup(
    name=__package_name__,
    version=__version__,
    license=__license__,
    description="",
    author=__author__,
    packages=find_packages("src"),
    package_dir={"": "src"},
    py_modules=[splitext(basename(path))[0] for path in glob("src/*.py")],
    include_package_data=True,
    zip_safe=False,
    python_requires=">=3.10`",
    install_requires=[
        # eg: 'aspectlib==1.1.1', 'six>=1.7',
    ],
    extras_require={
        "dev": ["pytest>=6.2.3"],
    },
)
