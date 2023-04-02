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

import six

from keras_tuner import protos
from keras_tuner.api_export import keras_tuner_export
from keras_tuner.engine import conditions as conditions_mod
from keras_tuner.engine.hyperparameters import hyperparameter


@keras_tuner_export("keras_tuner.engine.hyperparameters.Fixed")
class Fixed(hyperparameter.HyperParameter):
    """Fixed, untunable value.

    Args:
        name: A string. the name of parameter. Must be unique for each
            `HyperParameter` instance in the search space.
        value: The value to use (can be any JSON-serializable Python type).
    """

    def __init__(self, name, value, **kwargs):
        super().__init__(name=name, default=value, **kwargs)
        self.name = name

        if isinstance(value, bool):
            value = bool(value)
        elif isinstance(value, six.integer_types):
            value = int(value)
        elif isinstance(value, six.string_types):
            value = str(value)
        elif not isinstance(value, (float, str)):
            raise ValueError(
                "`Fixed` value must be an `int`, `float`, `str`, "
                f"or `bool`, found {value}"
            )
        self.value = value

    def __repr__(self):
        return f"Fixed(name: {self.name}, value: {self.value})"

    @property
    def values(self):
        return (self.value,)

    def prob_to_value(self, prob):
        return self.value

    def value_to_prob(self, value):
        return 0.5

    @property
    def default(self):
        return self.value

    def get_config(self):
        config = super().get_config()
        config["name"] = self.name
        config.pop("default")
        config["value"] = self.value
        return config

    @classmethod
    def from_proto(cls, proto):
        value = getattr(proto.value, proto.value.WhichOneof("kind"))
        conditions = [
            conditions_mod.Condition.from_proto(c) for c in proto.conditions
        ]
        return cls(name=proto.name, value=value, conditions=conditions)

    def to_proto(self):
        if isinstance(self.value, bool):
            # Check bool first as bool is subclass of int.
            # So bool is also six.integer_types.
            value = protos.get_proto().Value(boolean_value=self.value)
        elif isinstance(self.value, six.integer_types):
            value = protos.get_proto().Value(int_value=self.value)
        elif isinstance(self.value, float):
            value = protos.get_proto().Value(float_value=self.value)
        elif isinstance(self.value, six.string_types):
            value = protos.get_proto().Value(string_value=self.value)

        return protos.get_proto().Fixed(
            name=self.name,
            value=value,
            conditions=[c.to_proto() for c in self.conditions],
        )
