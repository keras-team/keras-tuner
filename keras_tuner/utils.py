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
"""KerasTuner utilities."""


import warnings

import tensorflow as tf
from packaging.version import parse

# Check if we are in a ipython/colab environement
try:
    import IPython

    class_name = IPython.get_ipython().__class__.__name__
    IS_NOTEBOOK = "Terminal" not in class_name
except (NameError, ImportError):  # pragma: no cover
    IS_NOTEBOOK = False  # pragma: no cover


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
    if parse(tf.__version__) < parse("2.0.0"):
        warnings.warn(
            "The Tensorflow package version needs to be at least 2.0.0 \n"
            "for AutoKeras to run. Currently, your TensorFlow version is \n"
            f"{tf.__version__}. Please upgrade with \n"
            "`$ pip install --upgrade tensorflow`. \n"
            "You can use `pip freeze` to check afterwards that everything is "
            "ok.",
            ImportWarning,
        )


def serialize_keras_object(obj):
    if hasattr(tf.keras.utils, "legacy"):
        return tf.keras.utils.legacy.serialize_keras_object(  # pragma: no cover
            obj
        )
    else:
        return tf.keras.utils.serialize_keras_object(obj)  # pragma: no cover


def deserialize_keras_object(config, module_objects=None, custom_objects=None):
    if hasattr(tf.keras.utils, "legacy"):
        return (  # pragma: no cover
            tf.keras.utils.legacy.deserialize_keras_object(
                config, custom_objects, module_objects
            )
        )
    else:
        return tf.keras.utils.deserialize_keras_object(  # pragma: no cover
            config, custom_objects, module_objects
        )


def to_list(values):
    if isinstance(values, list):
        return values
    if isinstance(values, tuple):
        return list(values)
    return [values]
