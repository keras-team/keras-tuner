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

from tensorflow import keras
import numpy as np


class HyperParameter(object):
    """HyperParameter base class.

    Args:
        name: Str. Name of parameter. Must be unique.
        default: Default value to return for the
            parameter.
    """

    def __init__(self, name, default=None):
        self.name = name
        self.default = default

    def get_config(self):
        return {'name': self.name, 'default': self.default}

    @property
    def default_value(self):
        return self.default

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
        default: Default value to return for the
            parameter.
    """

    def __init__(self, name, values, default=None):
        super(Choice, self).__init__(name=name, default=default)
        if not values:
            raise ValueError('`values` must be provided.')
        self.values = values

    @property
    def default_value(self):
        if self.default is not None:
            return self.default
        return self.values[0]

    def random_sample(self, seed=None):
        random_state = np.random.RandomState(seed)
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
        default: Default value to return for the
            parameter.
    """

    def __init__(self, name, min_value, max_value, step=1, default=None):
        super(Range, self).__init__(name=name, default=default)
        self.max_value = max_value
        self.min_value = min_value
        self.step = step
        self._values = list(range(min_value, max_value, step))

    def random_sample(self, seed=None):
        random_state = np.random.RandomState(seed)
        return random_state.choice(self._values)

    @property
    def default_value(self):
        if self.default is not None:
            return self.default
        return self.min_value

    def get_config(self):
        config = super(Range, self).get_config()
        config['max_value'] = self.max_value
        config['min_value'] = self.min_value
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
        default: Default value to return for the
            parameter.
    """

    def __init__(self, name, min_value, max_value, resolution, default=None):
        super(Logarithmic, self).__init__(name=name, default=default)
        self.max_value = max_value
        self.min_value = min_value
        self.resolution = resolution

    @property
    def default_value(self):
        if self.default is not None:
            return self.default
        return self.min_value

    def random_sample(self, seed):
        random_state = np.random.RandomState(seed)
        width = max_value - min_value
        value = self.min_value + random_state.random() * width
        quantized_value = round(value / self.resolution) * self.resolution
        return quantized_value

    def get_config(self):
        config = super(Range, self).get_config()
        config['max_value'] = self.max_value
        config['min_value'] = self.min_value
        config['num_bins'] = self.num_bins
        return config


class HyperParameters(object):
    """Container for both a hyperparameter space, and current values.

    Attributes:
        space: A list of HyperParameter instances.
        values: A dict mapping hyperparameter names to current values.
    """

    def __init__(self):
        self.space = []
        self.values = {}

    def retrieve(self, name, type, config):
        if name in self.values:
            # TODO: type compatibility check.
            return self.values[name]
        return self.register(name, type, config)

    def register(self, name, type, config):
        config['name'] = name
        config = {'class_name': type, 'config': config}
        p = deserialize(config)
        self.space.append(p)
        value = p.default_value
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

    def Range(self, name, min_value, max_value, step=None, default=None):
        return self.retrieve(name, 'Range',
                             config={'min_value': min_value,
                                     'max_value': max_value,
                                     'step': step,
                                     'default': default})

    def Linear(self, name, min_value, max_value, resolution=None, default=None):
        return self.retrieve(name, 'Range',
                             config={'min_value': min_value,
                                     'max_value': max_value,
                                     'resolution': resolution,
                                     'default': default})

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


def deserialize(config):
    module_objects = globals()
    return keras.utils.deserialize_keras_object(
        config, module_objects=module_objects)
