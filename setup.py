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

from setuptools import find_packages
from setuptools import setup

__version__ = "1.4.0"

setup(
    name="keras-tuner",
    description="A Hyperparameter Tuning Library for Keras",
    url="https://github.com/keras-team/keras-tuner",
    author="The KerasTuner authors",
    license="Apache License 2.0",
    version=__version__,
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
