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
        values: List of possible values. Values must be int, float,
            str, or bool. All values must be of the same type.
        ordered: Whether the values passed should be considered to
            have an ordering. This defaults to `True` for float/int
            values. Must be `False` for any other values.
        default: Default value to return for the parameter.
            If unspecified, the default value will be:
            - None if None is one of the choices in `values`
            - The first entry in `values` otherwise.
    """

    def __init__(self, name, values, ordered=None, default=None):
        super(Choice, self).__init__(name=name, default=default)
        if not values:
            raise ValueError('`values` must be provided.')
        self.values = values

        # Type checking.
        types = set(type(v) for v in values)
        unsupported_types = types - {int, float, str, bool}
        if unsupported_types:
            raise TypeError(
                'A `Choice` can contain only `int`, `float`, `str`, or '
                '`bool`, found values: ' + str(values) + 'with '
                'types: ' + str(unsupported_types))

        if len(types) > 1:
            raise TypeError(
                'A `Choice` can contain only one type of value, found '
                'values: ' + str(values) + ' with types ' + str(types))
        self._type = types.pop()

        # Get or infer ordered.
        self.ordered = ordered
        orderable_types = {int, float}
        if self.ordered and self._type not in orderable_types:
            raise ValueError('`ordered` must be `False` for non-numeric '
                             'types.')
        if self.ordered is None:
            self.ordered = self._type in orderable_types

        if default is not None and default not in values:
            raise ValueError(
                'The default value should be one of the choices. '
                'You passed: values=%s, default=%s' % (values, default))

    def __repr__(self):
        return (f'Choice(name: {self.name!r}, values: {self.values},'
                f' ordered: {self.ordered}, default: {self.default})')

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
        config['ordered'] = self.ordered
        return config


class Int(HyperParameter):
    """Integer range.

    Args:
        name: Str. Name of parameter. Must be unique.
        min_value: Int. Lower limit of range (included).
        max_value: Int. Upper limit of range (excluded).
        step: Int. Step of range.
        default: Default value to return for the parameter.
            If unspecified, the default value will be
            `min_value`.
    """

    def __init__(self, name, min_value, max_value, step=1, default=None):
        super(Int, self).__init__(name=name, default=default)
        self.max_value = int(max_value)
        self.min_value = int(min_value)
        self.step = int(step)
        self._values = list(range(min_value, max_value, step))

    def __repr__(self):
        return (f'Int(name: {self.name!r}, min_value: {self.min_value},'
                f' max_value: {self.max_value}, step: {self.step},'
                f' default: {self.default})')

    def random_sample(self, seed=None):
        random_state = random.Random(seed)
        return random_state.choice(self._values)

    @property
    def default(self):
        if self._default is not None:
            return self._default
        return self.min_value

    def get_config(self):
        config = super(Int, self).get_config()
        config['min_value'] = self.min_value
        config['max_value'] = self.max_value
        config['step'] = self.step
        config['default'] = self._default
        return config


class Float(HyperParameter):
    """Floating point range, can be evenly divided.

    Args:
        name: Str. Name of parameter. Must be unique.
        min_value: Float. Lower bound of the range.
        max_value: Float. Upper bound of the range.
        step: Optional. Float, e.g. 0.1.
            smallest meaningful distance between two values.
            Whether step should be specified is Oracle dependent,
            since some Oracles can infer an optimal step automatically.
        default: Default value to return for the parameter.
            If unspecified, the default value will be
            `min_value`.
    """

    def __init__(self, name, min_value, max_value, step=None, default=None):
        super(Float, self).__init__(name=name, default=default)
        self.max_value = float(max_value)
        self.min_value = float(min_value)
        if step is not None:
            self.step = float(step)
        else:
            self.step = None

    def __repr__(self):
        return (f'Float(name: {self.name!r}, min_value: {self.min_value},'
                f' max_value: {self.max_value}, step: {self.step},'
                f' default: {self.default})')

    @property
    def default(self):
        if self._default is not None:
            return self._default
        return self.min_value

    def random_sample(self, seed=None):
        random_state = random.Random(seed)
        if self.step is not None:
            width = self.max_value - self.min_value
            value = self.min_value + float(random_state.random()) * width
            quantized_value = round(value / self.step) * self.step
            return quantized_value
        else:
            return random_state.uniform(self.min_value, self.max_value)

    def get_config(self):
        config = super(Float, self).get_config()
        config['min_value'] = self.min_value
        config['max_value'] = self.max_value
        config['step'] = self.step
        return config


class Boolean(HyperParameter):
    """Choice between True and False.

    Args:
        name: Str. Name of parameter. Must be unique.
        default: Default value to return for the parameter.
            If unspecified, the default value will be False.
    """

    def __init__(self, name, default=False):
        super(Boolean, self).__init__(name=name, default=default)
        if default not in {True, False}:
            raise ValueError(
                '`default` must be a Python boolean. '
                'You passed: default=%s' % (default,))

    def __repr__(self):
        return (f'Boolean(name: {self.name!r}, '
                f' default: {self.default})')

    def random_sample(self, seed=None):
        random_state = random.Random(seed)
        return random_state.choice((True, False))


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
        return f'Fixed(name: {self.name!r}, value: {self.value})'

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
        self._scopes = []

    @contextlib.contextmanager
    def name_scope(self, name):
        self._scopes.append(name)
        try:
            yield
        finally:
            self._scopes.pop()

    @contextlib.contextmanager
    def conditional_scope(self, parent_name, parent_values):
        """Opens a scope to create conditional HyperParameters.

        All HyperParameters created under this scope will only be active
        when the parent HyperParameter specified by `parent_name` is
        equal to one of the values passed in `parent_values`.

        When the condition is not met, creating a HyperParameter under
        this scope will register the HyperParameter, but will return
        `None` rather than a concrete value.

        Note that any Python code under this scope will execute
        regardless of whether the condition is met.

        Arguments:
          parent_name: The name of the HyperParameter to condition on.
          parent_values: Values of the parent HyperParameter for which
            HyperParameters under this scope should be considered valid.
        """
        full_parent_name = self._get_name(parent_name)
        if full_parent_name not in self.values:
            raise ValueError(
                '`HyperParameter` named: ' + full_parent_name + ' '
                'not defined.')

        if not isinstance(parent_values, (list, tuple)):
            parent_values = [parent_values]

        parent_values = [str(v) for v in parent_values]

        self._scopes.append({'parent_name': parent_name,
                             'parent_values': parent_values})
        try:
            yield
        finally:
            self._scopes.pop()

    def _conditions_are_active(self, scopes=None):
        if scopes is None:
            scopes = self._scopes

        partial_scopes = []
        for scope in scopes:
            if self._is_conditional_scope(scope):
                full_name = self._get_name(
                    scope['parent_name'],
                    partial_scopes)
                if str(self.values[full_name]) not in scope['parent_values']:
                    return False
            partial_scopes.append(scope)
        return True

    def _retrieve(self,
                  name,
                  type,
                  config,
                  parent_name=None,
                  parent_values=None):
        """Gets or creates a `HyperParameter`."""
        if parent_name:
            with self.conditional_scope(parent_name, parent_values):
                return self._retrieve_helper(name, type, config)
        return self._retrieve_helper(name, type, config)

    def _retrieve_helper(self, name, type, config):
        self._check_name_is_valid(name)
        full_name = self._get_name(name)

        if full_name in self.values:
            # TODO: type compatibility check,
            # or name collision check.
            retrieved_value = self.values[full_name]
        else:
            retrieved_value = self.register(name, type, config)

        if self._conditions_are_active():
            return retrieved_value
        # Sanity check that a conditional HP that is not currently active
        # is not being inadvertently relied upon in the model building
        # function.
        return None

    def register(self, name, type, config):
        full_name = self._get_name(name)
        config['name'] = full_name
        config = {'class_name': type, 'config': config}
        p = deserialize(config)
        self.space.append(p)
        value = p.default
        self.values[full_name] = value
        return value

    def get(self, name):
        """Return the current value of this HyperParameter."""

        # Most common case: not attempting to access a conditional parent
        # or conditional child.
        full_name = self._get_name(name)
        if full_name in self.values:
            if self._conditions_are_active():
                return self.values[full_name]
            else:
                # Sanity check for conditional HP usage.
                return None

        # Check parent/child conditions.
        found_inactive = False
        # Remove conditional scopes from name.
        full_name_no_cond = '/'.join([
            p for p in self._get_name_parts(full_name) if
            not self._is_conditional_scope(p)])
        for hp_name in self.values.keys():
            hp_parts = self._get_name_parts(hp_name)
            # Remove conditional scopes from name.
            hp_name_no_cond = '/'.join(
                [p for p in hp_parts
                 if not self._is_conditional_scope(p)])
            if hp_name_no_cond == full_name_no_cond:
                # Check that this HP is active for this Trial.
                if self._conditions_are_active(hp_parts):
                    return self.values[hp_name]
                else:
                    found_inactive = True

        if found_inactive:
            # Sanity check, found only inactive conditional HPs.
            return None

        raise ValueError(
            'Unknown parameter: {}'.format(full_name_no_cond))

    def Choice(self,
               name,
               values,
               ordered=None,
               default=None,
               parent_name=None,
               parent_values=None):
        return self._retrieve(name, 'Choice',
                              config={'values': values,
                                      'ordered': ordered,
                                      'default': default},
                              parent_name=parent_name,
                              parent_values=parent_values)

    def Int(self,
            name,
            min_value,
            max_value,
            step=1,
            default=None,
            parent_name=None,
            parent_values=None):
        return self._retrieve(name, 'Int',
                              config={'min_value': min_value,
                                      'max_value': max_value,
                                      'step': step,
                                      'default': default},
                              parent_name=parent_name,
                              parent_values=parent_values)

    def Float(self,
              name,
              min_value,
              max_value,
              step=None,
              default=None,
              parent_name=None,
              parent_values=None):
        return self._retrieve(name, 'Float',
                              config={'min_value': min_value,
                                      'max_value': max_value,
                                      'step': step,
                                      'default': default},
                              parent_name=parent_name,
                              parent_values=parent_values)

    def Boolean(self,
                name,
                default=False,
                parent_name=None,
                parent_values=None):
        return self._retrieve(name, 'Boolean',
                              config={'default': default},
                              parent_name=parent_name,
                              parent_values=parent_values)

    def Fixed(self,
              name,
              value,
              parent_name=None,
              parent_values=None):
        return self._retrieve(name, 'Fixed',
                              config={'value': value},
                              parent_name=parent_name,
                              parent_values=parent_values)

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

    def _get_name(self, name, scopes=None):
        """Returns a name qualified by `name_scopes`."""
        if scopes is None:
            scopes = self._scopes

        scope_strings = []
        for scope in scopes:
            if self._is_name_scope(scope):
                scope_strings.append(scope)
            elif self._is_conditional_scope(scope):
                parent_name = scope['parent_name']
                parent_values = scope['parent_values']
                scope_string = '{name}={vals}'.format(
                    name=parent_name,
                    vals=','.join([str(val) for val in parent_values]))
                scope_strings.append(scope_string)
        return '/'.join(scope_strings + [name])

    def _get_name_parts(self, full_name):
        """Splits `full_name` into its scopes and leaf name."""
        str_parts = full_name.split('/')
        parts = []

        for part in str_parts:
            if '=' in part:
                parent_name, parent_values = part.split('=')
                parent_values = parent_values.split(',')
                parts.append({'parent_name': parent_name,
                              'parent_values': parent_values})
            else:
                parts.append(part)

        return parts

    def _check_name_is_valid(self, name):
        if '/' in name or '=' in name or ',' in name:
            raise ValueError(
                '`HyperParameter` names cannot contain "/", "=" or "," ' 
                'characters.')

        for scope in self._scopes[::-1]:
            if self._is_conditional_scope(scope):
                if name == scope['parent_name']:
                    raise ValueError(
                        'A conditional `HyperParameter` cannot have the same '
                        'name as its parent. Found: ' + str(name) + ' and '
                        'parent_name: ' + str(parent_name))
            else:
                # Names only have to be unique up to the last `name_scope`.
                break

    def _is_name_scope(self, scope):
        return isinstance(scope, str)

    def _is_conditional_scope(self, scope):
        return (isinstance(scope, dict) and
                'parent_name' in scope and 'parent_values' in scope)


def deserialize(config):
    module_objects = globals()
    return keras.utils.deserialize_keras_object(
        config, module_objects=module_objects)


HyperParameters.Choice.__doc__ == Choice.__doc__
HyperParameters.Int.__doc__ == Int.__doc__
HyperParameters.Float.__doc__ == Float.__doc__
HyperParameters.Fixed.__doc__ == Fixed.__doc__
