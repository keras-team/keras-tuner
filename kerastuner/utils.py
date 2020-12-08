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

import IPython
from packaging.version import parse
import tensorflow as tf


# Check if we are in a ipython/colab environement
try:
    class_name = IPython.get_ipython().__class__.__name__
    if "Terminal" in class_name:
        IS_NOTEBOOK = False
    else:
        IS_NOTEBOOK = True

except NameError:
    IS_NOTEBOOK = False


if IS_NOTEBOOK:
    from IPython import display


def try_clear():
    if IS_NOTEBOOK:
        display.clear_output()
    else:
        print()


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
            'The Tensorflow package version needs to be at least 2.0.0 \n'
            'for AutoKeras to run. Currently, your TensorFlow version is \n'
            '{version}. Please upgrade with \n'
            '`$ pip install --upgrade tensorflow`. \n'
            'You can use `pip freeze` to check afterwards that everything is '
            'ok.'.format(version=tf.__version__)
        )
