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
"HyperParameters logic."


import abc

import six

from keras_tuner import protos
from keras_tuner import utils


@six.add_metaclass(abc.ABCMeta)
class Condition:
    """Abstract condition for a conditional hyperparameter.

    Subclasses of this object can be passed to a `HyperParameter` to specify
    that this condition must be met for the hyperparameter to be active for the
    `Trial`.
    """

    @abc.abstractmethod
    def is_active(self, values):
        """Whether this condition should be considered active.

        Determines whether this condition is true for the current `Trial`.

        Args:
            values: Dict. The active values for this `Trial`. Keys are the
                names of the hyperparameters.

        Returns:
            A boolean value of whether the condition is true.
        """
        raise NotImplementedError("Must be implemented in subclasses.")

    @abc.abstractmethod
    def __eq__(self, other):
        raise NotImplementedError("Must be implemented in subclasses.")

    @abc.abstractmethod
    def get_config(self):
        raise NotImplementedError("Must be implemented in subclasses.")

    @classmethod
    def from_config(cls, config):
        return cls(**config)  # pytype: disable=not-instantiable

    @classmethod
    def from_proto(cls, proto):
        kind = proto.WhichOneof("kind")
        if kind == "parent":
            parent = getattr(proto, kind)
            name = parent.name
            values = parent.values
            values = [getattr(v, v.WhichOneof("kind")) for v in values]
            return Parent(name=name, values=values)
        raise ValueError(f"Unrecognized condition of type: {kind}")


class Parent(Condition):
    """Condition checking a `HyperParameter`'s value is in a list of values.

    It specifies a condition that a `HyperParameter`'s value is in a list of
    values. It can be used as the condition to activate another
    `HyperParameter` for the `Trial`.

    Example:

    ```python
    a = Choice('model', ['linear', 'dnn'])
    condition = Parent(name='model', value=['dnn'])
    b = Int('num_layers', 5, 10, conditions=[condition])
    ```

    Args:
        name: A string, the name of the `HyperParameter` to use in the
            condition.
        values: A list of values of the `HyperParameter` to activate the
            condition.
    """

    def __init__(self, name, values):
        self.name = name

        # Standardize on str, int, float, bool.
        values = utils.to_list(values)
        first_val = values[0]
        if isinstance(first_val, six.string_types):
            values = [str(v) for v in values]
        elif isinstance(first_val, bool):
            # Bool check needs to be before integer check to prevent bool falls
            # into integer condition.
            pass
        elif isinstance(first_val, six.integer_types):
            values = [int(v) for v in values]
        elif not isinstance(first_val, float):
            raise TypeError(
                "Can contain only `int`, `float`, `str`, or "
                "`bool`, found values: " + str(values) + "with "
                "types: " + str(type(first_val))
            )
        self.values = values

    def is_active(self, values):
        return self.name in values and values[self.name] in self.values

    def __eq__(self, other):
        return (
            isinstance(other, Parent)
            and other.name == self.name
            and other.values == self.values
        )

    def get_config(self):
        return {"name": self.name, "values": self.values}

    def to_proto(self):
        print(self.values[0])
        if isinstance(self.values[0], six.string_types):
            values = [
                protos.get_proto().Value(string_value=v) for v in self.values
            ]
        elif isinstance(self.values[0], bool):
            values = [
                protos.get_proto().Value(boolean_value=v) for v in self.values
            ]
        elif isinstance(self.values[0], six.integer_types):
            values = [
                protos.get_proto().Value(int_value=v) for v in self.values
            ]
        else:
            values = [
                protos.get_proto().Value(float_value=v) for v in self.values
            ]

        return protos.get_proto().Condition(
            parent=protos.get_proto().Condition.Parent(
                name=self.name, values=values
            )
        )


OBJECTS = (
    Condition,
    Parent,
)
ALL_CLASSES = {cls.__name__: cls for cls in OBJECTS}


def deserialize(config):
    return utils.deserialize_keras_object(config, module_objects=ALL_CLASSES)


def serialize(obj):
    return utils.serialize_keras_object(obj)
