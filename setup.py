"""Setup script."""
from __future__ import absolute_import

from setuptools import find_packages
from setuptools import setup
import time

version = "0.7.0"
stub = str(int(time.time()))  # Used to have increased version automagically
version = version + '.' + stub

setup(
    name="Kerastuner",
    version=version,
    description="Hypertuner for Keras",
    author='Elie Bursztein',
    author_email='elieb@google.com',
    url='https://fixme',
    license='Apache License 2.0',
    install_requires=[
        "attrs",
        "numpy",
        "pathlib",
        "tabulate",
        "termcolor",
        "tqdm"
    ],
    packages=find_packages()
)
