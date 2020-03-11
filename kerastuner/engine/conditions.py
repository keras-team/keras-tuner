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
"HyperParameters logic."

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import six

from ..protos import kerastuner_pb2


@six.add_metaclass(abc.ABCMeta)
class Condition(object):
    """Abstract condition for a conditional hyperparameter.

    Subclasses of this object can be passed to a `HyperParameter` to
    specify that this condition must be met in order for that hyperparameter
    to be considered active for the `Trial`.

    Example:

    ```

    a = Choice('model', ['linear', 'dnn'])
    condition = kt.conditions.Parent(name='a', value=['dnn'])
    b = Int('num_layers', 5, 10, conditions=[condition])
    ```
    """

    @abc.abstractmethod
    def is_active(self, values):
        """Whether this condition should be considered active.

        Determines whether this condition is true for the current `Trial`.

        # Arguments:
            values: Dict. The active values for this `Trial`. Keys are the
               names of the hyperparameters.

        # Returns:
            bool.
        """
        raise NotImplementedError('Must be implemented in subclasses.')

    @abc.abstractmethod
    def __eq__(self, other):
        raise NotImplementedError('Must be implemented in subclasses.')

    @abc.abstractmethod
    def get_config(self):
        raise NotImplementedError('Must be implemented in subclasses.')

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    @classmethod
    def from_proto(self, proto):
        kind = proto.WhichOneof('kind')
        if kind == 'parent':
            parent = getattr(proto, kind)
            name = parent.name
            values = parent.values
            values = [getattr(v, v.WhichOneof('kind')) for v in values]
            return Parent(name=name, values=values)
        raise ValueError('Unrecognized condition of type: {}'.format(kind))


class Parent(Condition):
    """Condition that checks a value is equal to one of a list of values.

    This object can be passed to a `HyperParameter` to specify that this
    condition must be met in order for that hyperparameter to be considered
    active for the `Trial`.

    Example:

    ```
    a = Choice('model', ['linear', 'dnn'])
    b = Int('num_layers', 5, 10, conditions=[kt.conditions.Parent('a', ['dnn'])])
    ```

    # Arguments:
        name: The name of a `HyperParameter`.
        values: Values for which the `HyperParameter` this object is
            passed to should be considered active.
    """
    def __init__(self, name, values):
        self.name = name

        # Standardize on str, int, float, bool.
        values = _to_list(values)
        first_val = values[0]
        if isinstance(first_val, six.string_types):
            values = [str(v) for v in values]
        elif isinstance(first_val, six.integer_types):
            values = [int(v) for v in values]
        elif not isinstance(first_val, (bool, float)):
            raise TypeError(
                'Can contain only `int`, `float`, `str`, or '
                '`bool`, found values: ' + str(values) + 'with '
                'types: ' + str(type(first_val)))
        self.values = values

    def is_active(self, values):
        return (self.name in values and values[self.name] in self.values)

    def __eq__(self, other):
        return (isinstance(other, Parent) and
                other.name == self.name and
                other.values == self.values)

    def get_config(self):
        return {'name': self.name,
                'values': self.values}

    def to_proto(self):
        if isinstance(self.values[0], six.string_types):
            values = [kerastuner_pb2.Value(string_value=v) for v in self.values]
        elif isinstance(self.values[0], six.integer_types):
            values = [kerastuner_pb2.Value(int_value=v) for v in self.values]
        else:
            values = [kerastuner_pb2.Value(float_value=v) for v in self.values]

        return kerastuner_pb2.Condition(
            parent=kerastuner_pb2.Condition.Parent(
                name=self.name,
                values=values))


def _to_list(values):
    if isinstance(values, list):
        return values
    if isinstance(values, tuple):
        return list(values)
    return [values]
