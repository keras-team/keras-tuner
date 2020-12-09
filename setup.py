# Copyright 2019 The Keras Tuner Authors
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

from __future__ import absolute_import

from setuptools import find_packages
from setuptools import setup

version = '1.0.3'

setup(
    name='keras-tuner',
    version=version,
    description='Hypertuner for Keras',
    url='https://github.com/keras-team/keras-tuner',
    author='The Keras Tuner authors',
    author_email='kerastuner@google.com',
    license='Apache License 2.0',
    # tensorflow isn't a dependency because it would force the
    # download of the gpu version or the cpu version.
    # users should install it manually.
    install_requires=[
        'packaging',
        'future',
        'numpy',
        'tabulate',
        'terminaltables',
        'colorama',
        'tqdm',
        'requests',
        'scipy',
        'scikit-learn',
        'tensorboard',
        'ipython'
    ],
    extras_require={
        'tests': ['pytest',
                  'flake8',
                  'mock',
                  'portpicker',
                  'pytest-xdist',
                  'pytest-cov'],
    },
    classifiers=[
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.6',
        'Operating System :: Unix',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: MacOS',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',
        'Topic :: Software Development'
    ],
    packages=find_packages(exclude=('tests',))
)
