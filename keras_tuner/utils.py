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

from keras_tuner import backend
from keras_tuner.backend import keras

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
    if not backend.io.exists(path):
        backend.io.makedirs(path)

    # If it does exist, and remove_existing is specified,
    # the directory will be removed and recreated.
    elif remove_existing:
        backend.io.rmtree(path)
        backend.io.makedirs(path)


def serialize_keras_object(obj):
    if hasattr(keras.utils, "legacy"):
        return keras.utils.legacy.serialize_keras_object(  # pragma: no cover
            obj
        )
    else:
        return keras.utils.serialize_keras_object(obj)  # pragma: no cover


def deserialize_keras_object(config, module_objects=None, custom_objects=None):
    if hasattr(keras.utils, "legacy"):
        return keras.utils.legacy.deserialize_keras_object(  # pragma: no cover
            config, custom_objects, module_objects
        )
    else:
        return keras.utils.deserialize_keras_object(  # pragma: no cover
            config, custom_objects, module_objects
        )


def to_list(values):
    if isinstance(values, list):
        return values
    if isinstance(values, tuple):
        return list(values)
    return [values]
