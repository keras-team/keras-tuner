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
from keras_tuner.engine.hyperparameters import hp_utils
from keras_tuner.engine.hyperparameters.hp_types import numerical
from keras_tuner.protos import keras_tuner_pb2


def _check_int(val, arg):
    int_val = int(val)
    if int_val != val:
        raise ValueError(
            f"{arg} must be an int, Received: {str(val)} of type {type(val)}."
        )
    return int_val


class Int(numerical.Numerical):
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
        step: Optional integer, the distance between two consecutive samples in the
            range. If left unspecified, it is possible to sample any integers in
            the interval. If `sampling="linear"`, it will be the minimum additve
            between two samples. If `sampling="log"`, it will be the minimum
            multiplier between two samples.
        sampling: String. One of "linear", "log", "reverse_log". Defaults to
            "linear". When sampling value, it always start from a value in range
            [0.0, 1.0). The `sampling` argument decides how the value is
            projected into the range of [min_value, max_value].
            "linear": min_value + value * (max_value - min_value)
            "log": min_value * (max_value / min_value) ^ value
            "reverse_log":
                max_value - min_value * ((max_value / min_value) ^ (1 - value) - 1)
        default: Integer, default value to return for the parameter. If
            unspecified, the default value will be `min_value`.
    """

    def __init__(
        self,
        name,
        min_value,
        max_value,
        step=None,
        sampling="linear",
        default=None,
        **kwargs,
    ):
        if step is not None:
            step = _check_int(step, arg="step")
        elif sampling == "linear":
            step = 1
        super().__init__(
            name=name,
            min_value=_check_int(min_value, arg="min_value"),
            max_value=_check_int(max_value, arg="max_value"),
            step=step,
            sampling=sampling,
            default=default,
            **kwargs,
        )

    def __repr__(self):
        return (
            f"Int(name: '{self.name}', min_value: {self.min_value}, "
            f"max_value: {self.max_value}, step: {self.step}, "
            f"sampling: {self.sampling}, default: {self.default})"
        )

    def prob_to_value(self, prob):
        if self.step is None:
            # prob is in range [0.0, 1.0), use max_value + 1 so that
            # max_value may be sampled.
            return int(self._sample_numerical_value(prob, self.max_value + 1))
        return int(self._sample_with_step(prob))

    def value_to_prob(self, value):
        if self.step is None:
            return self._numerical_to_prob(
                # + 0.5 to center the prob
                value + 0.5,
                # + 1 to include the max_value
                self.max_value + 1,
            )
        return self._to_prob_with_step(value)

    @property
    def default(self):
        return self._default if self._default is not None else self.min_value

    def get_config(self):
        config = super().get_config()
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
            step=proto.step or None,
            sampling=hp_utils.sampling_from_proto(proto.sampling),
            default=proto.default,
            conditions=conditions,
        )

    def to_proto(self):
        return keras_tuner_pb2.Int(
            name=self.name,
            min_value=self.min_value,
            max_value=self.max_value,
            step=self.step if self.step is not None else 0,
            sampling=hp_utils.sampling_to_proto(self.sampling),
            default=self.default,
            conditions=[c.to_proto() for c in self.conditions],
        )
