# Copyright 2019 The KerasTuner Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Setup script."""

import os
import pathlib

from setuptools import find_packages
from setuptools import setup


def read(rel_path):
    here = os.path.abspath(os.path.dirname(__file__))
    with open(os.path.join(here, rel_path)) as fp:
        return fp.read()


def get_version(rel_path):
    for line in read(rel_path).splitlines():
        if line.startswith("__version__"):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    raise RuntimeError("Unable to find version string.")


HERE = pathlib.Path(__file__).parent
README = (HERE / "README.md").read_text()
if os.path.exists("keras_tuner/version.py"):
    VERSION = get_version("keras_tuner/version.py")
else:
    VERSION = get_version("keras_tuner/__init__.py")

setup(
    name="keras-tuner",
    description="A Hyperparameter Tuning Library for Keras",
    long_description_content_type="text/markdown",
    long_description=README,
    url="https://github.com/keras-team/keras-tuner",
    author="The KerasTuner authors",
    license="Apache License 2.0",
    version=VERSION,
    install_requires=[
        "keras-core",
        "packaging",
        "requests",
        "kt-legacy",
    ],
    extras_require={
        "tensorflow": [
            "tensorflow>=2.0",
        ],
        "tensorflow-cpu": [
            "tensorflow-cpu>=2.0",
        ],
        "tests": [
            "black",
            "flake8",
            "isort",
            "ipython",
            "pandas",
            "portpicker",
            "pytest",
            "pytest-cov",
            "pytest-xdist",
            "namex",
            "scikit-learn",
            "scipy",
        ],
        "bayesian": [
            "scikit-learn",
            "scipy",
        ],
        "build": [
            "tensorflow-cpu",
            "namex",
            "build",
        ],
    },
    classifiers=[
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: Unix",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: MacOS",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "Topic :: Software Development",
    ],
    packages=find_packages(exclude=("*test*",)),
)
