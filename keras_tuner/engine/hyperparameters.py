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


import collections
import contextlib
import copy
import math
import random

import numpy as np
import six
from tensorflow import keras

from keras_tuner.engine import conditions as conditions_mod
from keras_tuner.protos import keras_tuner_pb2


def _check_sampling_arg(sampling, step, min_value, max_value, hp_type="int"):
    if sampling is None:
        return None
    if min_value > max_value:
        raise ValueError(
            "`sampling` `min_value` "
            + str(min_value)
            + " is greater than the `max_value` "
            + str(max_value)
        )
    if hp_type == "int" and step != 1:
        raise ValueError("`sampling` can only be set on an `Int` when `step=1`.")
    if hp_type != "int" and step is not None:
        raise ValueError(
            "`sampling` and `step` cannot both be set, found "
            "`sampling`: " + str(sampling) + ", `step`: " + str(step)
        )

    _sampling_values = {"linear", "log", "reverse_log"}
    sampling = sampling.lower()
    if sampling not in _sampling_values:
        raise ValueError("`sampling` must be one of " + str(_sampling_values))
    if sampling in {"log", "reverse_log"} and min_value <= 0:
        raise ValueError(
            '`sampling="' + str(sampling) + '" is not supported for '
            "negative values, found `min_value`: " + str(min_value)
        )
    return sampling


def _check_int(val, arg):
    int_val = int(val)
    if int_val != val:
        raise ValueError(arg + " must be an int, found: " + str(val))
    return int_val


class HyperParameter(object):
    """Hyperparameter base class.

    A `HyperParameter` instance is uniquely identified by its `name` and
    `conditions` attributes. `HyperParameter`s with the same `name` but with
    different `conditions` are considered as different `HyperParameter`s by
    the `HyperParameters` instance.

    Args:
        name: A string. the name of parameter. Must be unique for each
            `HyperParameter` instance in the search space.
        default: The default value to return for the parameter.
        conditions: A list of `Condition`s for this object to be considered
            active.
    """

    def __init__(self, name, default=None, conditions=None):
        self.name = name
        self._default = default

        conditions = _to_list(conditions) if conditions else []
        self.conditions = [deserialize(c) for c in conditions]

    def get_config(self):
        conditions = [serialize(c) for c in self.conditions]
        return {"name": self.name, "default": self.default, "conditions": conditions}

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
        super(Choice, self).__init__(name=name, default=default, **kwargs)
        if not values:
            raise ValueError("`values` must be provided.")

        # Type checking.
        types = set(type(v) for v in values)
        if len(types) > 1:
            raise TypeError(
                "A `Choice` can contain only one type of value, found "
                "values: " + str(values) + " with types " + str(types)
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
        self.values = values

        if default is not None and default not in values:
            raise ValueError(
                "The default value should be one of the choices. "
                "You passed: values=%s, default=%s" % (values, default)
            )
        self._default = default

        # Get or infer ordered.
        self.ordered = ordered
        is_numeric = isinstance(values[0], (six.integer_types, float))
        if self.ordered and not is_numeric:
            raise ValueError("`ordered` must be `False` for non-numeric " "types.")
        if self.ordered is None:
            self.ordered = is_numeric

    def __repr__(self):
        return 'Choice(name: "{}", values: {}, ordered: {}, default: {})'.format(
            self.name, self.values, self.ordered, self.default
        )

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
        config["values"] = self.values
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


class Int(HyperParameter):
    """Integer range.

    Note that unlike Python's `range` function, `max_value` is *included* in
    the possible values this parameter can take on.

    Args:
        name: A string. the name of parameter. Must be unique for each
            `HyperParameter` instance in the search space.
        min_value: Integer, the lower limit of range, inclusive.
        max_value: Integer, the upper limit of range, inclusive.
        step: Integer, the distance between two consecutive samples in the
            range. Defaults to 1.
        sampling: Optional string. One of "linear", "log", "reverse_log". Acts
            as a hint for an initial prior probability distribution for how
            this value should be sampled, e.g. "log" will assign equal
            probabilities to each order of magnitude range.
        default: Integer, default value to return for the parameter. If
            unspecified, the default value will be `min_value`.
    """

    def __init__(
        self,
        name,
        min_value,
        max_value,
        step=1,
        sampling=None,
        default=None,
        **kwargs
    ):
        super(Int, self).__init__(name=name, default=default, **kwargs)
        self.max_value = _check_int(max_value, arg="max_value")
        self.min_value = _check_int(min_value, arg="min_value")
        self.step = _check_int(step, arg="step")
        self.sampling = _check_sampling_arg(
            sampling, step, min_value, max_value, hp_type="int"
        )

    def __repr__(self):
        return (
            'Int(name: "{}", min_value: {}, max_value: {}, step: {}, '
            "sampling: {}, default: {})"
        ).format(
            self.name,
            self.min_value,
            self.max_value,
            self.step,
            self.sampling,
            self.default,
        )

    def random_sample(self, seed=None):
        random_state = random.Random(seed)
        prob = float(random_state.random())
        return cumulative_prob_to_value(prob, self)

    @property
    def default(self):
        if self._default is not None:
            return self._default
        return self.min_value

    def get_config(self):
        config = super(Int, self).get_config()
        config["min_value"] = self.min_value
        config["max_value"] = self.max_value
        config["step"] = self.step
        config["sampling"] = self.sampling
        config["default"] = self._default
        return config

    @classmethod
    def from_proto(cls, proto):
        conditions = [
            conditions_mod.Condition.from_proto(c) for c in proto.conditions
        ]
        return cls(
            name=proto.name,
            min_value=proto.min_value,
            max_value=proto.max_value,
            step=proto.step if proto.step else None,
            sampling=_sampling_from_proto(proto.sampling),
            default=proto.default,
            conditions=conditions,
        )

    def to_proto(self):
        return keras_tuner_pb2.Int(
            name=self.name,
            min_value=self.min_value,
            max_value=self.max_value,
            step=self.step if self.step is not None else 0,
            sampling=_sampling_to_proto(self.sampling),
            default=self.default,
            conditions=[c.to_proto() for c in self.conditions],
        )


class Float(HyperParameter):
    """Floating point range, can be evenly divided.

    Args:
        name: A string. the name of parameter. Must be unique for each
            `HyperParameter` instance in the search space.
        min_value: Float, the lower bound of the range.
        max_value: Float, the upper bound of the range.
        step: Optional float, e.g. 0.1, the smallest meaningful distance
            between two values. Whether step should be specified is Oracle
            dependent, since some Oracles can infer an optimal step
            automatically.
        sampling: Optional string. One of "linear", "log", "reverse_log". Acts
            as a hint for an initial prior probability distribution for how
            this value should be sampled, e.g. "log" will assign equal
            probabilities to each order of magnitude range.
        default: Float, the default value to return for the parameter. If
            unspecified, the default value will be `min_value`.
    """

    def __init__(
        self,
        name,
        min_value,
        max_value,
        step=None,
        sampling=None,
        default=None,
        **kwargs
    ):
        super(Float, self).__init__(name=name, default=default, **kwargs)
        self.max_value = float(max_value)
        self.min_value = float(min_value)
        if step is not None:
            self.step = float(step)
        else:
            self.step = None
        self.sampling = _check_sampling_arg(
            sampling, step, min_value, max_value, hp_type="float"
        )

    def __repr__(self):
        return (
            'Float(name: "{}", min_value: {}, max_value: {}, step: {}, '
            "sampling: {}, default: {})"
        ).format(
            self.name,
            self.min_value,
            self.max_value,
            self.step,
            self.sampling,
            self.default,
        )

    @property
    def default(self):
        if self._default is not None:
            return self._default
        return self.min_value

    def random_sample(self, seed=None):
        random_state = random.Random(seed)
        prob = float(random_state.random())
        return cumulative_prob_to_value(prob, self)

    def get_config(self):
        config = super(Float, self).get_config()
        config["min_value"] = self.min_value
        config["max_value"] = self.max_value
        config["step"] = self.step
        config["sampling"] = self.sampling
        return config

    @classmethod
    def from_proto(cls, proto):
        conditions = [
            conditions_mod.Condition.from_proto(c) for c in proto.conditions
        ]
        return cls(
            name=proto.name,
            min_value=proto.min_value,
            max_value=proto.max_value,
            step=proto.step if proto.step else None,
            sampling=_sampling_from_proto(proto.sampling),
            default=proto.default,
            conditions=conditions,
        )

    def to_proto(self):
        return keras_tuner_pb2.Float(
            name=self.name,
            min_value=self.min_value,
            max_value=self.max_value,
            step=self.step if self.step is not None else 0.0,
            sampling=_sampling_to_proto(self.sampling),
            default=self.default,
            conditions=[c.to_proto() for c in self.conditions],
        )


class Boolean(HyperParameter):
    """Choice between True and False.

    Args:
        name: A string. the name of parameter. Must be unique for each
            `HyperParameter` instance in the search space.
        default: Boolean, the default value to return for the parameter.
            If unspecified, the default value will be False.
    """

    def __init__(self, name, default=False, **kwargs):
        super(Boolean, self).__init__(name=name, default=default, **kwargs)
        if default not in {True, False}:
            raise ValueError(
                "`default` must be a Python boolean. "
                "You passed: default=%s" % (default,)
            )

    def __repr__(self):
        return 'Boolean(name: "{}", default: {})'.format(self.name, self.default)

    def random_sample(self, seed=None):
        random_state = random.Random(seed)
        return random_state.choice((True, False))

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


class Fixed(HyperParameter):
    """Fixed, untunable value.

    Args:
        name: A string. the name of parameter. Must be unique for each
            `HyperParameter` instance in the search space.
        value: The value to use (can be any JSON-serializable Python type).
    """

    def __init__(self, name, value, **kwargs):
        super(Fixed, self).__init__(name=name, default=value, **kwargs)
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
                "or `bool`, found {}".format(value)
            )
        self.value = value

    def __repr__(self):
        return "Fixed(name: {}, value: {})".format(self.name, self.value)

    def random_sample(self, seed=None):
        return self.value

    @property
    def default(self):
        return self.value

    def get_config(self):
        config = super(Fixed, self).get_config()
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
        if isinstance(self.value, six.integer_types):
            value = keras_tuner_pb2.Value(int_value=self.value)
        elif isinstance(self.value, float):
            value = keras_tuner_pb2.Value(float_value=self.value)
        elif isinstance(self.value, six.string_types):
            value = keras_tuner_pb2.Value(string_value=self.value)
        else:
            value = keras_tuner_pb2.Value(boolean_value=self.value)

        return keras_tuner_pb2.Fixed(
            name=self.name,
            value=value,
            conditions=[c.to_proto() for c in self.conditions],
        )


class HyperParameters(object):
    """Container for both a hyperparameter space, and current values.

    A `HyperParameters` instance can be pass to `HyperModel.build(hp)` as an
    argument to build a model.

    Attributes:
        values: A dict mapping hyperparameter names to current values.
    """

    def __init__(self):
        # Current name scopes.
        self._name_scopes = []
        # Current `Condition`s, managed by `conditional_scope`.
        self._conditions = []

        # Dict of list of hyperparameters with same
        # name but different conditions, e.g. `{name: [hp1, hp2]}`.
        # Hyperparameters are uniquely identified by their name and
        # conditions.
        self._hps = collections.defaultdict(list)

        # List of hyperparameters, maintained in insertion order.
        # This guarantees that conditional params are always later in
        # the list than their parents.
        self._space = []

        # Active values for this `Trial`.
        self.values = {}

        # A list of active `conditional_scope`s in a build,
        # each of which is a list of condtions.
        self.active_scopes = []
        # Similar for inactive `conditional_scope`s.
        self.inactive_scopes = []

    @contextlib.contextmanager
    def name_scope(self, name):
        self._name_scopes.append(name)
        try:
            yield
        finally:
            self._name_scopes.pop()

    @contextlib.contextmanager
    def conditional_scope(self, parent_name, parent_values):
        """Opens a scope to create conditional HyperParameters.

        All `HyperParameter`s created under this scope will only be active when
        the parent `HyperParameter` specified by `parent_name` is equal to one
        of the values passed in `parent_values`.

        When the condition is not met, creating a `HyperParameter` under this
        scope will register the `HyperParameter`, but will return `None` rather
        than a concrete value.

        Note that any Python code under this scope will execute regardless of
        whether the condition is met.

        This feature is for the `Tuner` to collect more information of the
        search space and the current trial.  It is especially useful for model
        selection. If the parent `HyperParameter` is for model selection, the
        `HyperParameter`s in a model should only be active when the model
        selected, which can be implemented using `conditional_scope`.

        Examples:

        ```python
        def MyHyperModel(HyperModel):
            def build(self, hp):
                model = Sequential()
                model.add(Input(shape=(32, 32, 3)))
                model_type = hp.Choice("model_type", ["mlp", "cnn"])
                with hp.conditional_scope("model_type", ["mlp"]):
                    if model_type == "mlp":
                        model.add(Flatten())
                        model.add(Dense(32, activation='relu'))
                with hp.conditional_scope("model_type", ["cnn"]):
                    if model_type == "cnn":
                        model.add(Conv2D(64, 3, activation='relu'))
                        model.add(GlobalAveragePooling2D())
                model.add(Dense(10, activation='softmax'))
                return model
        ```

        Args:
            parent_name: A string, specifying the name of the parent
                `HyperParameter` to use as the condition to activate the
                current `HyperParameter`.
            parent_values: A list of the values of the parent `HyperParameter`
                to use as the condition to activate the current
                `HyperParameter`.
        """
        parent_name = self._get_name(parent_name)  # Add name_scopes.
        if not self._exists(parent_name):
            raise ValueError(
                "`HyperParameter` named: " + parent_name + " " "not defined."
            )

        condition = conditions_mod.Parent(parent_name, parent_values)
        self._conditions.append(condition)

        if condition.is_active(self.values):
            self.active_scopes.append(copy.deepcopy(self._conditions))
        else:
            self.inactive_scopes.append(copy.deepcopy(self._conditions))

        try:
            yield
        finally:
            self._conditions.pop()

    def is_active(self, hyperparameter):
        """Checks if a hyperparameter is currently active for a `Trial`.

        A hyperparameter is considered active if and only if all its parent
        conditions are active, and not affected by whether the hyperparameter
        is used while building the model. The function is usually called by the
        `Oracle` for populating new hyperparameter values and updating the trial
        after receiving the evaluation results.

        Args:
            hp: A string or `HyperParameter` instance. If string, checks whether
                any hyperparameter with that name is active. If `HyperParameter`
                instance, checks whether the object is active.

        Returns:
            A boolean, whether the hyperparameter is active.
        """
        hp = hyperparameter
        if isinstance(hp, HyperParameter):
            return self._conditions_are_active(hp.conditions)
        hp_name = str(hp)
        for temp_hp in self._hps[hp_name]:
            if self._conditions_are_active(temp_hp.conditions):
                return True
        return False

    def _conditions_are_active(self, conditions=None):
        if conditions is None:
            conditions = self._conditions

        for condition in conditions:
            if not condition.is_active(self.values):
                return False
        return True

    def _exists(self, name, conditions=None):
        """Checks for a hyperparameter with the same name and conditions."""
        if conditions is None:
            conditions = self._conditions

        if name in self._hps:
            hps = self._hps[name]
            for hp in hps:
                if hp.conditions == conditions:
                    return True
        return False

    def _retrieve(self, hp):
        """Gets or creates a hyperparameter.

        Args:
            hp: A `HyperParameter` instance.

        Returns:
            The value of the hyperparameter, or None if the hyperparameter is
            not active.
        """
        if self._exists(hp.name, hp.conditions):
            if self._conditions_are_active(hp.conditions):
                return self.values[hp.name]
            return None  # Ensures inactive values are not relied on by user.
        return self._register(hp)

    def _register(self, hyperparameter, overwrite=False):
        """Registers a hyperparameter in this container.

        Args:
            hp: A `HyperParameter` instance.
            overwrite: Boolean, whether to overwrite the existing value with
                the default hyperparameter value.

        Returns:
            The value of the hyperparameter, or None if the hyperparameter is
            not active.
        """
        hp = hyperparameter
        # Copy to ensure this param can be serialized.
        hp = hp.__class__.from_config(hp.get_config())
        self._hps[hp.name].append(hp)
        self._space.append(hp)
        # Only add active values to `self.values`.
        if self._conditions_are_active(hp.conditions):
            # Use the default value only if not populated.
            if overwrite or hp.name not in self.values:
                self.values[hp.name] = hp.default
            return self.values[hp.name]
        return None  # Ensures inactive values are not relied on by user.

    def get(self, name):
        """Return the current value of this hyperparameter set."""
        name = self._get_name(name)  # Add name_scopes.
        if name in self.values:
            return self.values[name]  # Only active values are added here.
        elif name in self._hps:
            raise ValueError("{} is currently inactive.".format(name))
        else:
            raise KeyError("{} does not exist.".format(name))

    def __getitem__(self, name):
        return self.get(name)

    def __contains__(self, name):
        try:
            self.get(name)
            return True
        except (KeyError, ValueError):
            return False

    def Choice(
        self,
        name,
        values,
        ordered=None,
        default=None,
        parent_name=None,
        parent_values=None,
    ):
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
            parent_name: Optional string, specifying the name of the parent
                `HyperParameter` to use as the condition to activate the
                current `HyperParameter`.
            parent_values: Optional list of the values of the parent
                `HyperParameter` to use as the condition to activate the
                current `HyperParameter`.

        Returns:
            The value of the hyperparameter, or None if the hyperparameter is
            not active.
        """
        with self._maybe_conditional_scope(parent_name, parent_values):
            hp = Choice(
                name=self._get_name(name),  # Add name_scopes.
                values=values,
                ordered=ordered,
                default=default,
                conditions=self._conditions,
            )
            return self._retrieve(hp)

    def Int(
        self,
        name,
        min_value,
        max_value,
        step=1,
        sampling=None,
        default=None,
        parent_name=None,
        parent_values=None,
    ):
        """Integer range.

        Note that unlike Python's `range` function, `max_value` is *included* in
        the possible values this parameter can take on.

        Args:
            name: A string. the name of parameter. Must be unique for each
                `HyperParameter` instance in the search space.
            min_value: Integer, the lower limit of range, inclusive.
            max_value: Integer, the upper limit of range, inclusive.
            step: Integer, the distance between two consecutive samples in the
                range. Defaults to 1.
            sampling: Optional string. One of "linear", "log", "reverse_log". Acts
                as a hint for an initial prior probability distribution for how
                this value should be sampled, e.g. "log" will assign equal
                probabilities to each order of magnitude range.
            default: Integer, default value to return for the parameter. If
                unspecified, the default value will be `min_value`.
            parent_name: Optional string, specifying the name of the parent
                `HyperParameter` to use as the condition to activate the
                current `HyperParameter`.
            parent_values: Optional list of the values of the parent
                `HyperParameter` to use as the condition to activate the
                current `HyperParameter`.

        Returns:
            The value of the hyperparameter, or None if the hyperparameter is
            not active.
        """
        with self._maybe_conditional_scope(parent_name, parent_values):
            hp = Int(
                name=self._get_name(name),  # Add name_scopes.
                min_value=min_value,
                max_value=max_value,
                step=step,
                sampling=sampling,
                default=default,
                conditions=self._conditions,
            )
            return self._retrieve(hp)

    def Float(
        self,
        name,
        min_value,
        max_value,
        step=None,
        sampling=None,
        default=None,
        parent_name=None,
        parent_values=None,
    ):
        """Floating point range, can be evenly divided.

        Args:
            name: A string. the name of parameter. Must be unique for each
                `HyperParameter` instance in the search space.
            min_value: Float, the lower bound of the range.
            max_value: Float, the upper bound of the range.
            step: Optional float, e.g. 0.1, the smallest meaningful distance
                between two values. Whether step should be specified is Oracle
                dependent, since some Oracles can infer an optimal step
                automatically.
            sampling: Optional string. One of "linear", "log", "reverse_log". Acts
                as a hint for an initial prior probability distribution for how
                this value should be sampled, e.g. "log" will assign equal
                probabilities to each order of magnitude range.
            default: Float, the default value to return for the parameter. If
                unspecified, the default value will be `min_value`.
            parent_name: Optional string, specifying the name of the parent
                `HyperParameter` to use as the condition to activate the
                current `HyperParameter`.
            parent_values: Optional list of the values of the parent
                `HyperParameter` to use as the condition to activate the
                current `HyperParameter`.

        Returns:
            The value of the hyperparameter, or None if the hyperparameter is
            not active.
        """
        with self._maybe_conditional_scope(parent_name, parent_values):
            hp = Float(
                name=self._get_name(name),  # Add name_scopes.
                min_value=min_value,
                max_value=max_value,
                step=step,
                sampling=sampling,
                default=default,
                conditions=self._conditions,
            )
            return self._retrieve(hp)

    def Boolean(self, name, default=False, parent_name=None, parent_values=None):
        """Choice between True and False.

        Args:
            name: A string. the name of parameter. Must be unique for each
                `HyperParameter` instance in the search space.
            default: Boolean, the default value to return for the parameter.
                If unspecified, the default value will be False.
            parent_name: Optional string, specifying the name of the parent
                `HyperParameter` to use as the condition to activate the
                current `HyperParameter`.
            parent_values: Optional list of the values of the parent
                `HyperParameter` to use as the condition to activate the
                current `HyperParameter`.

        Returns:
            The value of the hyperparameter, or None if the hyperparameter is
            not active.
        """
        with self._maybe_conditional_scope(parent_name, parent_values):
            hp = Boolean(
                name=self._get_name(name),  # Add name_scopes.
                default=default,
                conditions=self._conditions,
            )
            return self._retrieve(hp)

    def Fixed(self, name, value, parent_name=None, parent_values=None):
        """Fixed, untunable value.

        Args:
            name: A string. the name of parameter. Must be unique for each
                `HyperParameter` instance in the search space.
            value: The value to use (can be any JSON-serializable Python type).
            parent_name: Optional string, specifying the name of the parent
                `HyperParameter` to use as the condition to activate the
                current `HyperParameter`.
            parent_values: Optional list of the values of the parent
                `HyperParameter` to use as the condition to activate the
                current `HyperParameter`.

        Returns:
            The value of the hyperparameter, or None if the hyperparameter is
            not active.
        """
        with self._maybe_conditional_scope(parent_name, parent_values):
            hp = Fixed(
                name=self._get_name(name),  # Add name_scopes.
                value=value,
                conditions=self._conditions,
            )
            return self._retrieve(hp)

    @property
    def space(self):
        return self._space

    def get_config(self):
        return {
            "space": [
                {"class_name": p.__class__.__name__, "config": p.get_config()}
                for p in self.space
            ],
            "values": dict((k, v) for (k, v) in self.values.items()),
        }

    @classmethod
    def from_config(cls, config):
        hps = cls()
        for p in config["space"]:
            p = deserialize(p)
            hps._hps[p.name].append(p)
            hps._space.append(p)
        hps.values = dict((k, v) for (k, v) in config["values"].items())
        return hps

    def copy(self):
        return HyperParameters.from_config(self.get_config())

    def merge(self, hps, overwrite=True):
        """Merges hyperparameters into this object.

        Args:
            hps: A `HyperParameters` object or list of `HyperParameter`
                objects.
            overwrite: Boolean, whether existing `HyperParameter`s should be
                overridden by those in `hps` with the same name and conditions.
        """
        if isinstance(hps, HyperParameters):
            hps = hps.space

        if not overwrite:
            hps = [hp for hp in hps if not self._exists(hp.name, hp.conditions)]

        for hp in hps:
            self._register(hp, overwrite)

    @classmethod
    def from_proto(cls, proto):
        hps = cls()

        space = []
        if isinstance(proto, keras_tuner_pb2.HyperParameters.Values):
            # Allows passing in only values, space becomes `Fixed`.
            for name, value in proto.values.items():
                space.append(Fixed(name, getattr(value, value.WhichOneof("kind"))))
        else:
            for fixed_proto in proto.space.fixed_space:
                space.append(Fixed.from_proto(fixed_proto))
            for float_proto in proto.space.float_space:
                space.append(Float.from_proto(float_proto))
            for int_proto in proto.space.int_space:
                space.append(Int.from_proto(int_proto))
            for choice_proto in proto.space.choice_space:
                space.append(Choice.from_proto(choice_proto))
            for boolean_proto in proto.space.boolean_space:
                space.append(Boolean.from_proto(boolean_proto))

        hps.merge(space)

        if isinstance(proto, keras_tuner_pb2.HyperParameters.Values):
            values = proto.values
        else:
            values = proto.values.values
        for name, val in values.items():
            hps.values[name] = getattr(val, val.WhichOneof("kind"))

        return hps

    def to_proto(self):
        fixed_space = []
        float_space = []
        int_space = []
        choice_space = []
        boolean_space = []
        for hp in self.space:
            if isinstance(hp, Fixed):
                fixed_space.append(hp.to_proto())
            elif isinstance(hp, Float):
                float_space.append(hp.to_proto())
            elif isinstance(hp, Int):
                int_space.append(hp.to_proto())
            elif isinstance(hp, Choice):
                choice_space.append(hp.to_proto())
            elif isinstance(hp, Boolean):
                boolean_space.append(hp.to_proto())
            else:
                raise ValueError("Unrecognized HP type: {}".format(hp))

        values = {}
        for name, value in self.values.items():
            if isinstance(value, float):
                val = keras_tuner_pb2.Value(float_value=value)
            elif isinstance(value, six.integer_types):
                val = keras_tuner_pb2.Value(int_value=value)
            elif isinstance(value, six.string_types):
                val = keras_tuner_pb2.Value(string_value=value)
            elif isinstance(value, bool):
                val = keras_tuner_pb2.Value(boolean_value=value)
            else:
                raise ValueError("Unrecognized value type: {}".format(value))
            values[name] = val

        return keras_tuner_pb2.HyperParameters(
            space=keras_tuner_pb2.HyperParameters.Space(
                fixed_space=fixed_space,
                float_space=float_space,
                int_space=int_space,
                choice_space=choice_space,
                boolean_space=boolean_space,
            ),
            values=keras_tuner_pb2.HyperParameters.Values(values=values),
        )

    @contextlib.contextmanager
    def _maybe_conditional_scope(self, parent_name, parent_values):
        if parent_name:
            with self.conditional_scope(parent_name, parent_values):
                yield
        else:
            yield

    def _get_name(self, name, name_scopes=None):
        """Returns a name qualified by `name_scopes`."""
        if name_scopes is None:
            name_scopes = self._name_scopes

        if name_scopes:
            return "/".join(name_scopes) + "/" + str(name)
        return str(name)

    def _validate_name(self, name):
        for condition in self._conditions:
            if condition.name == name:
                raise ValueError(
                    "A conditional `HyperParameter` cannot have the same "
                    "name as its parent. Found: " + str(name) + " and "
                    "parent_name: " + str(condition.name)
                )


def deserialize(config):
    # Autograph messes with globals(), so in order to support HPs inside `call` we
    # have to enumerate them manually here.
    objects = (
        HyperParameter,
        Fixed,
        Float,
        Int,
        Choice,
        Boolean,
        HyperParameters,
        conditions_mod.Condition,
        conditions_mod.Parent,
        int,
        float,
        str,
        bool,
    )
    if isinstance(config, objects):
        return config  # Already deserialized.
    module_objects = {cls.__name__: cls for cls in objects}
    return keras.utils.deserialize_keras_object(
        config, module_objects=module_objects
    )


def serialize(obj):
    if isinstance(obj, (int, float, str, bool)):
        return obj
    return keras.utils.serialize_keras_object(obj)


def cumulative_prob_to_value(prob, hp):
    """Convert a value from [0, 1] to a hyperparameter value."""
    if isinstance(hp, Fixed):
        return hp.value
    elif isinstance(hp, Boolean):
        return bool(prob >= 0.5)
    elif isinstance(hp, Choice):
        ele_prob = 1 / len(hp.values)
        index = int(math.floor(prob / ele_prob))
        # Can happen when `prob` is very close to 1.
        if index == len(hp.values):
            index = index - 1
        return hp.values[index]
    elif isinstance(hp, (Int, Float)):
        sampling = hp.sampling or "linear"
        if sampling == "linear":
            value = prob * (hp.max_value - hp.min_value) + hp.min_value
        elif sampling == "log":
            value = hp.min_value * math.pow(hp.max_value / hp.min_value, prob)
        elif sampling == "reverse_log":
            value = (
                hp.max_value
                + hp.min_value
                - hp.min_value * math.pow(hp.max_value / hp.min_value, 1 - prob)
            )
        else:
            raise ValueError("Unrecognized sampling value: {}".format(sampling))

        if hp.step is not None:
            values = np.arange(hp.min_value, hp.max_value + 1e-7, step=hp.step)
            closest_index = np.abs(values - value).argmin()
            value = values[closest_index]

        if isinstance(hp, Int):
            return int(value)
        return value
    else:
        raise ValueError("Unrecognized `HyperParameter` type: {}".format(hp))


def value_to_cumulative_prob(value, hp):
    """Convert a hyperparameter value to [0, 1]."""
    if isinstance(hp, Fixed):
        return 0.5
    if isinstance(hp, Boolean):
        # Center the value in its probability bucket.
        if value:
            return 0.75
        return 0.25
    elif isinstance(hp, Choice):
        ele_prob = 1 / len(hp.values)
        index = hp.values.index(value)
        # Center the value in its probability bucket.
        return (index + 0.5) * ele_prob
    elif isinstance(hp, (Int, Float)):
        if hp.max_value == hp.min_value:
            return 1.0
        sampling = hp.sampling or "linear"
        if sampling == "linear":
            return (value - hp.min_value) / (hp.max_value - hp.min_value)
        elif sampling == "log":
            return math.log(value / hp.min_value) / math.log(
                hp.max_value / hp.min_value
            )
        elif sampling == "reverse_log":
            return 1.0 - math.log(
                (hp.max_value + hp.min_value - value) / hp.min_value
            ) / math.log(hp.max_value / hp.min_value)
        else:
            raise ValueError("Unrecognized sampling value: {}".format(sampling))
    else:
        raise ValueError("Unrecognized `HyperParameter` type: {}".format(hp))


def _sampling_from_proto(sampling):
    if sampling is None or sampling == keras_tuner_pb2.Sampling.NONE:
        return None
    if sampling == keras_tuner_pb2.Sampling.LINEAR:
        return "linear"
    if sampling == keras_tuner_pb2.Sampling.LOG:
        return "log"
    if sampling == keras_tuner_pb2.Sampling.REVERSE_LOG:
        return "reverse_log"
    raise ValueError("Unrecognized sampling: {}".format(sampling))


def _sampling_to_proto(sampling):
    if sampling is None:
        return keras_tuner_pb2.Sampling.NONE
    if sampling == "linear":
        return keras_tuner_pb2.Sampling.LINEAR
    if sampling == "log":
        return keras_tuner_pb2.Sampling.LOG
    if sampling == "reverse_log":
        return keras_tuner_pb2.Sampling.REVERSE_LOG
    raise ValueError("Unrecognized sampling: {}".format(sampling))


def _to_list(values):
    if isinstance(values, list):
        return values
    if isinstance(values, tuple):
        return list(values)
    return [values]
