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

from keras_tuner.engine import conditions as conditions_mod
from keras_tuner.engine.hyperparameters import hyperparameter
from keras_tuner.protos import keras_tuner_pb2


class Boolean(hyperparameter.HyperParameter):
    """Choice between True and False.

    Args:
        name: A string. the name of parameter. Must be unique for each
            `HyperParameter` instance in the search space.
        default: Boolean, the default value to return for the parameter.
            If unspecified, the default value will be False.
    """

    def __init__(self, name, default=False, **kwargs):
        super().__init__(name=name, default=default, **kwargs)
        if default not in {True, False}:
            raise ValueError(
                f"`default` must be a Python boolean. You passed: default={default}"
            )

    def __repr__(self):
        return f'Boolean(name: "{self.name}", default: {self.default})'

    @property
    def values(self):
        return (True, False)

    def prob_to_value(self, prob):
        return bool(prob >= 0.5)

    def value_to_prob(self, value):
        # Center the value in its probability bucket.
        return 0.75 if value else 0.25

    @classmethod
    def from_proto(cls, proto):
        conditions = [
            conditions_mod.Condition.from_proto(c) for c in proto.conditions
        ]
        return cls(name=proto.name, default=proto.default, conditions=conditions)

    def to_proto(self):
        return keras_tuner_pb2.Boolean(
            name=self.name,
            default=self.default,
            conditions=[c.to_proto() for c in self.conditions],
        )
