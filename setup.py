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
import time

version = "0.9.0"
stub = str(int(time.time()))  # Used to increase version automagically.
version = version + '.' + stub

setup(
    name="Keras Tuner",
    version=version,
    description="Hypertuner for Keras",
    author='Elie Bursztein',
    author_email='kerastuner@google.com',
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
        "tabulate",
        "terminaltables",
        "termcolor",
        "colorama",
        "tqdm",
        "requests",
        "psutil",
        "sklearn"
    ],
    packages=find_packages()
)
