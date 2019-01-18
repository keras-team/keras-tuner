"""Setup script."""
from __future__ import absolute_import

from setuptools import find_packages
from setuptools import setup
import time

version = "0.7"
stub = str(int(time.time()))  # Used to increase version automagically.
version = version + '.' + stub

setup(
    name="Kerastuner",
    version=version,
    description="Hypertuner for Keras",
    author='Elie Bursztein',
    author_email='elieb@google.com',
    url='https://fixme',
    license='Apache License 2.0',
    entry_points='''
        [console_scripts]
        kerastuner-summary=kerastuner.tools.summary:summary
        kerastuner-status=kerastuner.tools.status:status
    ''',
    install_requires=[
        "art",
        "attrs",
        "etaprogress",
        "numpy",
        "pathlib",
        "tabulate",  # to be removed
        "terminaltables",
        "termcolor",  # to be removed
        "colorama",
        "tqdm",
        "requests"
    ],
    packages=find_packages()
)
