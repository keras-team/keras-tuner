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
"""Keras Tuner utilities."""

from packaging.version import parse
import tensorflow as tf


def create_directory(path, remove_existing=False):
    # Create the directory if it doesn't exist.
    if not tf.io.gfile.exists(path):
        tf.io.gfile.makedirs(path)

    # If it does exist, and remove_existing is specified,
    # the directory will be removed and recreated.
    elif remove_existing:
        tf.io.gfile.rmtree(path)
        tf.io.gfile.makedirs(path)


def check_tf_version():
    if parse(tf.__version__) < parse('2.0.0'):
        raise ImportError(
            f'The Tensorflow package version needs to be at least v2.0.0 \n'
            f'for Keras Tuner to run. Currently, your TensorFlow version is \n'
            f'v{tf.__version__}. Please upgrade with \n'
            f'`$ pip install --upgrade tensorflow` -> GPU version \n'
            f'or \n'
            f'`$ pip install --upgrade tensorflow-cpu` -> CPU version. \n'
            f'You can use `pip freeze` to check afterwards that everything is ok.'
        )
