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

from keras_tuner.engine import conditions as conditions_mod
from keras_tuner.engine.hyperparameters import hp_utils
from keras_tuner.engine.hyperparameters import hyperparameter
from keras_tuner.protos import keras_tuner_pb2


class Choice(hyperparameter.HyperParameter):
    """Choice of one value among a predefined set of possible values.

    Args:
        name: A string. the name of parameter. Must be unique for each
            `HyperParameter` instance in the search space.
        values: A list of possible values. Values must be int, float,
            str, or bool. All values must be of the same type.
        ordered: Optional boolean, whether the values passed should be
            considered to have an ordering. Defaults to `True` for float/int
            values.  Must be `False` for any other values.
        default: Optional default value to return for the parameter.
            If unspecified, the default value will be:
            - None if None is one of the choices in `values`
            - The first entry in `values` otherwise.
    """

    def __init__(self, name, values, ordered=None, default=None, **kwargs):
        super().__init__(name=name, default=default, **kwargs)
        if not values:
            raise ValueError("`values` must be provided for `Choice`.")

        # Type checking.
        types = {type(v) for v in values}
        if len(types) > 1:
            raise TypeError(
                "A `Choice` can contain only one type of value, "
                f"found values: {str(values)} with types {types}."
            )

        # Standardize on str, int, float, bool.
        if isinstance(values[0], six.string_types):
            values = [str(v) for v in values]
            if default is not None:
                default = str(default)
        elif isinstance(values[0], six.integer_types):
            values = [int(v) for v in values]
            if default is not None:
                default = int(default)
        elif not isinstance(values[0], (bool, float)):
            raise TypeError(
                "A `Choice` can contain only `int`, `float`, `str`, or "
                "`bool`, found values: " + str(values) + "with "
                "types: " + str(type(values[0]))
            )
        self._values = values

        if default is not None and default not in values:
            raise ValueError(
                "The default value should be one of the choices. "
                f"You passed: values={values}, default={default}"
            )
        self._default = default

        # Get or infer ordered.
        self.ordered = ordered
        is_numeric = isinstance(values[0], (six.integer_types, float))
        if self.ordered and not is_numeric:
            raise ValueError("`ordered` must be `False` for non-numeric types.")
        if self.ordered is None:
            self.ordered = is_numeric

    def __repr__(self):
        return (
            f"Choice(name: '{self.name}', "
            + f"values: {self._values}, "
            + f"ordered: {self.ordered}, default: {self.default})"
        )

    @property
    def values(self):
        return self._values

    @property
    def default(self):
        return self._values[0] if self._default is None else self._default

    def prob_to_value(self, prob):
        return self._values[hp_utils.prob_to_index(prob, len(self._values))]

    def value_to_prob(self, value):
        return hp_utils.index_to_prob(self._values.index(value), len(self._values))

    def get_config(self):
        config = super().get_config()
        config["values"] = self._values
        config["ordered"] = self.ordered
        return config

    @classmethod
    def from_proto(cls, proto):
        values = [getattr(val, val.WhichOneof("kind")) for val in proto.values]
        default = getattr(proto.default, proto.default.WhichOneof("kind"), None)
        conditions = [
            conditions_mod.Condition.from_proto(c) for c in proto.conditions
        ]
        return cls(
            name=proto.name,
            values=values,
            ordered=proto.ordered,
            default=default,
            conditions=conditions,
        )

    def to_proto(self):
        if isinstance(self.values[0], six.string_types):
            values = [keras_tuner_pb2.Value(string_value=v) for v in self.values]
            default = keras_tuner_pb2.Value(string_value=self.default)
        elif isinstance(self.values[0], six.integer_types):
            values = [keras_tuner_pb2.Value(int_value=v) for v in self.values]
            default = keras_tuner_pb2.Value(int_value=self.default)
        else:
            values = [keras_tuner_pb2.Value(float_value=v) for v in self.values]
            default = keras_tuner_pb2.Value(float_value=self.default)
        return keras_tuner_pb2.Choice(
            name=self.name,
            ordered=self.ordered,
            values=values,
            default=default,
            conditions=[c.to_proto() for c in self.conditions],
        )
