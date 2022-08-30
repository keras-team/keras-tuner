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

from tensorflow import keras

from keras_tuner.engine.hyperparameters import hps
from keras_tuner.engine.hyperparameters.hps import Boolean
from keras_tuner.engine.hyperparameters.hps import Choice
from keras_tuner.engine.hyperparameters.hps import Fixed
from keras_tuner.engine.hyperparameters.hps import Float
from keras_tuner.engine.hyperparameters.hps import Int
from keras_tuner.engine.hyperparameters.hyperparameter import HyperParameter
from keras_tuner.engine.hyperparameters.hyperparameters import HyperParameters

OBJECTS = hps.OBJECTS + (
    HyperParameter,
    HyperParameters,
)

ALL_CLASSES = {cls.__name__: cls for cls in OBJECTS}


def deserialize(config):
    objects = (
        int,
        float,
        str,
        bool,
    )
    # Remove the if block after resolving
    # https://github.com/keras-team/autokeras/issues/1765
    if isinstance(config, objects):
        return config
    return keras.utils.deserialize_keras_object(config, module_objects=ALL_CLASSES)


def serialize(obj):
    return keras.utils.serialize_keras_object(obj)
