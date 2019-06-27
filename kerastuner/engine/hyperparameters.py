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

import contextlib
import random

from tensorflow import keras


class HyperParameter(object):
    """HyperParameter base class.

    Args:
        name: Str. Name of parameter. Must be unique.
        default: Default value to return for the
            parameter.
    """

    def __init__(self, name, default=None):
        self.name = name
        self._default = default

    def get_config(self):
        return {'name': self.name, 'default': self.default}

    @property
    def default(self):
        return self._default

    def random_sample(self, seed=None):
        raise NotImplementedError

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class Choice(HyperParameter):
    """Choice of one value among a predefined set of possible values.

    Args:
        name: Str. Name of parameter. Must be unique.
        values: List of possible values.
            Any serializable type is allowed.
        default: Default value to return for the parameter.
            If unspecified, the default value will be:
            - None if None is one of the choices in `values`
            - The first entry in `values` otherwise.
    """

    def __init__(self, name, values, default=None):
        super(Choice, self).__init__(name=name, default=default)
        if not values:
            raise ValueError('`values` must be provided.')
        self.values = values
        if default is not None and default not in values:
            raise ValueError(
                'The default value should be one of the choices. '
                'You passed: values=%s, default=%s' % (values, default))

    def __repr__(self):
        return f"Choice(name: {self.name!r}, values: {self.values}"

    @property
    def default(self):
        if self._default is None:
            if None in self.values:
                return None
            return self.values[0]
        return self._default

    def random_sample(self, seed=None):
        random_state = random.Random(seed)
        return random_state.choice(self.values)

    def get_config(self):
        config = super(Choice, self).get_config()
        config['values'] = self.values
        return config


class Range(HyperParameter):
    """Integer range.

    Args:
        name: Str. Name of parameter. Must be unique.
        min_value: Int. Lower limit of range (included).
        max_value: Int. Upper limite of range (excluded).
        step: Int. Step of range.
        default: Default value to return for the parameter.
            If unspecified, the default value will be
            `min_value`.
    """

    def __init__(self, name, min_value, max_value, step=1, default=None):
        super(Range, self).__init__(name=name, default=default)
        self.max_value = int(max_value)
        self.min_value = int(min_value)
        self.step = int(step)
        self._values = list(range(min_value, max_value, step))

    def __repr__(self):
        return (f"Range(name: {self.name!r}, min_value: {self.min_value}," +
                f" max_value: {self.max_value}, step: {self.step})")

    def random_sample(self, seed=None):
        random_state = random.Random(seed)
        return random_state.choice(self._values)

    @property
    def default(self):
        if self._default is not None:
            return self._default
        return self.min_value

    def get_config(self):
        config = super(Range, self).get_config()
        config['min_value'] = self.min_value
        config['max_value'] = self.max_value
        config['step'] = self.step
        return config


class Linear(HyperParameter):
    """Floating point range, evenly divided.

    Args:
        name: Str. Name of parameter. Must be unique.
        min_value: Float. Lower bound of the range.
        max_value: Float. Upper bound of the range.
        resolution: Float, e.g. 0.1.
            smallest meaningful distance between two values.
        default: Default value to return for the parameter.
            If unspecified, the default value will be
            `min_value`.
    """

    def __init__(self, name, min_value, max_value, resolution, default=None):
        super(Linear, self).__init__(name=name, default=default)
        self.max_value = float(max_value)
        self.min_value = float(min_value)
        self.resolution = float(resolution)

    def __repr__(self):
        return (f"Linear(name: {self.name!r}, min_value: {self.min_value}," +
                f" max_value: {self.max_value}," +
                f" resolution: {self.resolution})")

    @property
    def default(self):
        if self._default is not None:
            return self._default
        return self.min_value

    def random_sample(self, seed=None):
        random_state = random.Random(seed)
        width = self.max_value - self.min_value
        value = self.min_value + float(random_state.random()) * width
        quantized_value = round(value / self.resolution) * self.resolution
        return quantized_value

    def get_config(self):
        config = super(Linear, self).get_config()
        config['min_value'] = self.min_value
        config['max_value'] = self.max_value
        config['resolution'] = self.resolution
        return config


class Fixed(HyperParameter):
    """Fixed, untunable value.

    Args:
        name: Str. Name of parameter. Must be unique.
        value: Value to use (can be any JSON-serializable
            Python type).
    """

    def __init__(self, name, value):
        self.name = name
        self.value = value

    def __repr__(self):
        return f"Fixed(name: {self.name!r}, value: {self.value}"

    def random_sample(self, seed=None):
        return self.value

    @property
    def default(self):
        return self.value

    def get_config(self):
        return {'name': self.name, 'value': self.value}


class HyperParameters(object):
    """Container for both a hyperparameter space, and current values.

    Attributes:
        space: A list of HyperParameter instances.
        values: A dict mapping hyperparameter names to current values.
    """

    def __init__(self):
        self.space = []
        self.values = {}
        self._name_scopes = []

    @contextlib.contextmanager
    def name_scope(self, name):
        self._name_scopes.append(name)
        try:
            yield
        finally:
            self._name_scopes.pop()

    def retrieve(self, name, type, config):
        full_name = self._get_full_name(name)
        if full_name in self.values:
            # TODO: type compatibility check,
            # or name collision check.
            return self.values[full_name]
        return self.register(full_name, type, config)

    def register(self, name, type, config):
        config['name'] = name
        config = {'class_name': type, 'config': config}
        p = deserialize(config)
        self.space.append(p)
        value = p.default
        self.values[name] = value
        return value

    def get(self, name):
        if name in self.values:
            return self.values[name]
        else:
            raise ValueError('Unknown parameter: {name}'.format(name=name))

    def Choice(self, name, values, default=None):
        return self.retrieve(name, 'Choice',
                             config={'values': values,
                                     'default': default})

    def Range(self, name, min_value, max_value, step=1, default=None):
        return self.retrieve(name, 'Range',
                             config={'min_value': min_value,
                                     'max_value': max_value,
                                     'step': step,
                                     'default': default})

    def Linear(self, name, min_value, max_value, resolution, default=None):
        return self.retrieve(name, 'Linear',
                             config={'min_value': min_value,
                                     'max_value': max_value,
                                     'resolution': resolution,
                                     'default': default})

    def Fixed(self, name, value):
        return self.retrieve(name, 'Fixed',
                             config={'value': value})

    def get_config(self):
        return {
            'space': [{'class_name': p.__class__.__name__,
                       'config': p.get_config()} for p in self.space],
            'values': dict((k, v) for (k, v) in self.values.items()),
        }

    @classmethod
    def from_config(cls, config):
        hp = cls()
        hp.space = [deserialize(p) for p in config['space']]
        hp.values = dict((k, v) for (k, v) in config['values'].items())
        return hp

    def copy(self):
        return HyperParameters.from_config(self.get_config())

    def _get_full_name(self, name):
        return '/'.join(self._name_scopes + [name])


def deserialize(config):
    module_objects = globals()
    return keras.utils.deserialize_keras_object(
        config, module_objects=module_objects)


HyperParameters.Choice.__doc__ == Choice.__doc__
HyperParameters.Range.__doc__ == Range.__doc__
HyperParameters.Linear.__doc__ == Linear.__doc__
HyperParameters.Fixed.__doc__ == Fixed.__doc__
