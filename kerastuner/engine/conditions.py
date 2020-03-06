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
    condition = kt.conditions.OneOf('a', 'dnn')

    a = Choice('model', ['linear', 'dnn'])
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


class OneOf(Condition):
    """Condition that checks a value is equal to one of a list of values.

    This object can be passed to a `HyperParameter` to specify that this
    condition must be met in order for that hyperparameter to be considered
    active for the `Trial`.

    Example:

    ```
    a = Choice('model', ['linear', 'dnn'])
    b = Int('num_layers', 5, 10, conditions=[kt.conditions.OneOf('a', ['dnn'])])
    ```

    # Arguments:
        name: The name of a `HyperParameter`.
        values: Values for which the `HyperParameter` this object is
            passed to should be considered active.
    """
    def __init__(self, name, values):
        self.name = name
        self.values = _to_list(values)

    def is_active(self, values):
        return (self.name in values and values[self.name] in self.values)

    def __eq__(self, other):
        return (isinstance(other, OneOf) and
                other.name == self.name and
                other.values == self.values)

    def get_config(self):
        return {'name': self.name,
                'values': self.values}


def _to_list(values):
    if isinstance(values, list):
        return values
    if isinstance(values, tuple):
        return list(values)
    return [values]
