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

import six

from keras_tuner.engine import conditions as conditions_mod
from keras_tuner.engine.hyperparameters import hp_types
from keras_tuner.engine.hyperparameters import hyperparameter as hp_module
from keras_tuner.protos import keras_tuner_pb2


class HyperParameters:
    """Container for both a hyperparameter space, and current values.

    A `HyperParameters` instance can be pass to `HyperModel.build(hp)` as an
    argument to build a model.

    To prevent the users from depending on inactive hyperparameter values, only
    active hyperparameters should have values in `HyperParameters.values`.

    Attributes:
        space: A list of `HyperParameter` objects.
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
        # Used by BaseTuner to activate all conditions.
        # No need to empty these after builds since when building the model, hp
        # is copied from the Oracle's hp, which always have these 2 fields empty.
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
            raise ValueError(f"`HyperParameter` named: {parent_name} not defined.")

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
        if isinstance(hp, hp_module.HyperParameter):
            return self._conditions_are_active(hp.conditions)
        hp_name = str(hp)
        return any(
            self._conditions_are_active(temp_hp.conditions)
            for temp_hp in self._hps[hp_name]
        )

    def _conditions_are_active(self, conditions):
        return all(condition.is_active(self.values) for condition in conditions)

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
            if self.is_active(hp):
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
        self._validate_name(hp.name)
        # Copy to ensure this param can be serialized.
        hp = hp.__class__.from_config(hp.get_config())
        self._hps[hp.name].append(hp)
        self._space.append(hp)
        # Only add active values to `self.values`.
        if self.is_active(hp):
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
            raise ValueError(f"{name} is currently inactive.")
        else:
            raise KeyError(f"{name} does not exist.")

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
            hp = hp_types.Choice(
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
        step=None,
        sampling="linear",
        default=None,
        parent_name=None,
        parent_values=None,
    ):
        """Integer hyperparameter.

        Note that unlike Python's `range` function, `max_value` is *included* in
        the possible values this parameter can take on.


        Example #1:

        ```py
        hp.Int(
            "n_layers",
            min_value=6,
            max_value=12)
        ```

        The possible values are [6, 7, 8, 9, 10, 11, 12].

        Example #2:

        ```py
        hp.Int(
            "n_layers",
            min_value=6,
            max_value=13,
            step=3)
        ```

        `step` is the minimum distance between samples.
        The possible values are [6, 9, 12].

        Example #3:

        ```py
        hp.Int(
            "batch_size",
            min_value=2,
            max_value=32,
            step=2,
            sampling="log")
        ```

        When `sampling="log"` the `step` is multiplied between samples.
        The possible values are [2, 4, 8, 16, 32].

        Args:
            name: A string. the name of parameter. Must be unique for each
                `HyperParameter` instance in the search space.
            min_value: Integer, the lower limit of range, inclusive.
            max_value: Integer, the upper limit of range, inclusive.
            step: Optional integer, the distance between two consecutive samples
                in the range. If left unspecified, it is possible to sample any
                integers in the interval. If `sampling="linear"`, it will be the
                minimum additve between two samples. If `sampling="log"`, it
                will be the minimum multiplier between two samples.
            sampling: String. One of "linear", "log", "reverse_log". Defaults to
                "linear". When sampling value, it always start from a value in
                range [0.0, 1.0). The `sampling` argument decides how the value
                is projected into the range of [min_value, max_value].
                "linear": min_value + value * (max_value - min_value)
                "log": min_value * (max_value / min_value) ^ value
                "reverse_log":
                    (max_value -
                     min_value * ((max_value / min_value) ^ (1 - value) - 1))
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
            hp = hp_types.Int(
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
        sampling="linear",
        default=None,
        parent_name=None,
        parent_values=None,
    ):
        """Floating point value hyperparameter.

        Example #1:

        ```py
        hp.Float(
            "image_rotation_factor",
            min_value=0,
            max_value=1)
        ```

        All values in interval [0, 1] have equal probability of being sampled.

        Example #2:

        ```py
        hp.Float(
            "image_rotation_factor",
            min_value=0,
            max_value=1,
            step=0.2)
        ```

        `step` is the minimum distance between samples.
        The possible values are [0, 0.2, 0.4, 0.6, 0.8, 1.0].

        Example #3:

        ```py
        hp.Float(
            "learning_rate",
            min_value=0.001,
            max_value=10,
            step=10,
            sampling="log")
        ```

        When `sampling="log"`, the `step` is multiplied between samples.
        The possible values are [0.001, 0.01, 0.1, 1, 10].

        Args:
            name: A string. the name of parameter. Must be unique for each
                `HyperParameter` instance in the search space.
            min_value: Float, the lower bound of the range.
            max_value: Float, the upper bound of the range.
            step: Optional float, the distance between two consecutive samples
                in the range. If left unspecified, it is possible to sample any
                value in the interval. If `sampling="linear"`, it will be the
                minimum additve between two samples. If `sampling="log"`, it
                will be the minimum multiplier between two samples.
            sampling: String. One of "linear", "log", "reverse_log". Defaults to
                "linear". When sampling value, it always start from a value in
                range [0.0, 1.0). The `sampling` argument decides how the value
                is projected into the range of [min_value, max_value].
                "linear": min_value + value * (max_value - min_value)
                "log": min_value * (max_value / min_value) ^ value
                "reverse_log":
                    (max_value -
                     min_value * ((max_value / min_value) ^ (1 - value) - 1))
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
            hp = hp_types.Float(
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
            hp = hp_types.Boolean(
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
            hp = hp_types.Fixed(
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
            "values": dict(self.values.items()),
        }

    @classmethod
    def from_config(cls, config):
        hps = cls()
        for p in config["space"]:
            p = hp_types.deserialize(p)
            hps._hps[p.name].append(p)
            hps._space.append(p)
        hps.values = dict(config["values"].items())
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

    def ensure_active_values(self):
        """Add and remove values if necessary.

        KerasTuner requires only active values to be populated. This function
        removes the inactive values and add the missing active values.

        Args:
            hps: HyperParameters, whose values to be ensured.
        """
        for hp in self.space:
            if self.is_active(hp):
                if hp.name not in self.values:
                    self.values[hp.name] = hp.random_sample()
            else:
                self.values.pop(hp.name, None)

    @classmethod
    def from_proto(cls, proto):
        hps = cls()

        space = []
        if isinstance(proto, keras_tuner_pb2.HyperParameters.Values):
            # Allows passing in only values, space becomes `Fixed`.
            space.extend(
                hp_types.Fixed(name, getattr(value, value.WhichOneof("kind")))
                for name, value in proto.values.items()
            )

        else:
            space.extend(
                hp_types.Fixed.from_proto(fixed_proto)
                for fixed_proto in proto.space.fixed_space
            )

            space.extend(
                hp_types.Float.from_proto(float_proto)
                for float_proto in proto.space.float_space
            )

            space.extend(
                hp_types.Int.from_proto(int_proto)
                for int_proto in proto.space.int_space
            )

            space.extend(
                hp_types.Choice.from_proto(choice_proto)
                for choice_proto in proto.space.choice_space
            )

            space.extend(
                hp_types.Boolean.from_proto(boolean_proto)
                for boolean_proto in proto.space.boolean_space
            )

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
            if isinstance(hp, hp_types.Fixed):
                fixed_space.append(hp.to_proto())
            elif isinstance(hp, hp_types.Float):
                float_space.append(hp.to_proto())
            elif isinstance(hp, hp_types.Int):
                int_space.append(hp.to_proto())
            elif isinstance(hp, hp_types.Choice):
                choice_space.append(hp.to_proto())
            elif isinstance(hp, hp_types.Boolean):
                boolean_space.append(hp.to_proto())
            else:
                raise ValueError(f"Unrecognized HP type: {hp}")

        values = {}
        for name, value in self.values.items():
            if isinstance(value, bool):
                val = keras_tuner_pb2.Value(boolean_value=value)
            elif isinstance(value, float):
                val = keras_tuner_pb2.Value(float_value=value)
            elif isinstance(value, six.integer_types):
                val = keras_tuner_pb2.Value(int_value=value)
            elif isinstance(value, six.string_types):
                val = keras_tuner_pb2.Value(string_value=value)
            else:
                raise ValueError(f"Unrecognized value type: {value}")
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

        return "/".join(name_scopes) + "/" + str(name) if name_scopes else str(name)

    def _validate_name(self, name):
        for condition in self._conditions:
            if condition.name == name:
                raise ValueError(
                    "A conditional `HyperParameter` cannot have the same "
                    "name as its parent. Found: " + str(name) + " and "
                    "parent_name: " + str(condition.name)
                )
