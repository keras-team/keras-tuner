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

from keras_tuner import utils
from keras_tuner.api_export import keras_tuner_export
from keras_tuner.backend import keras
from keras_tuner.engine.hyperparameters import hp_types
from keras_tuner.engine.hyperparameters.hp_types import Boolean
from keras_tuner.engine.hyperparameters.hp_types import Choice
from keras_tuner.engine.hyperparameters.hp_types import Fixed
from keras_tuner.engine.hyperparameters.hp_types import Float
from keras_tuner.engine.hyperparameters.hp_types import Int
from keras_tuner.engine.hyperparameters.hyperparameter import HyperParameter
from keras_tuner.engine.hyperparameters.hyperparameters import HyperParameters

OBJECTS = hp_types.OBJECTS + (
    HyperParameter,
    HyperParameters,
)

ALL_CLASSES = {cls.__name__: cls for cls in OBJECTS}


@keras_tuner_export("keras_tuner.engine.hyperparameters.deserialize")
def deserialize(config):
    return utils.deserialize_keras_object(config, module_objects=ALL_CLASSES)


@keras_tuner_export("keras_tuner.engine.hyperparameters.serialize")
def serialize(obj):
    return utils.serialize_keras_object(obj)
