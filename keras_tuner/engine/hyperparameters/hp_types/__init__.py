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

from keras_tuner.engine.hyperparameters.hp_types.boolean_hp import Boolean
from keras_tuner.engine.hyperparameters.hp_types.choice_hp import Choice
from keras_tuner.engine.hyperparameters.hp_types.fixed_hp import Fixed
from keras_tuner.engine.hyperparameters.hp_types.float_hp import Float
from keras_tuner.engine.hyperparameters.hp_types.int_hp import Int

OBJECTS = (
    Fixed,
    Float,
    Int,
    Choice,
    Boolean,
)

ALL_CLASSES = {cls.__name__: cls for cls in OBJECTS}


def deserialize(config):
    return keras.utils.deserialize_keras_object(config, module_objects=ALL_CLASSES)


def serialize(obj):
    return keras.utils.serialize_keras_object(obj)
