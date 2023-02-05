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


class Float(numerical.Numerical):
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
        step: Optional float, the distance between two consecutive samples in the
            range. If left unspecified, it is possible to sample any value in
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
        default: Float, the default value to return for the parameter. If
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
            self.step = float(step)
        super().__init__(
            name=name,
            min_value=float(min_value),
            max_value=float(max_value),
            step=step,
            sampling=sampling,
            default=default,
            **kwargs,
        )

    def __repr__(self):
        return (
            f"Float(name: '{self.name}', min_value: '{self.min_value}', "
            f"max_value: '{self.max_value}', step: '{self.step}', "
            f"sampling: '{self.sampling}', default: '{self.default}')"
        )

    @property
    def default(self):
        return self._default if self._default is not None else self.min_value

    def prob_to_value(self, prob):
        if self.step is None:
            return self._sample_numerical_value(prob)
        return self._sample_with_step(prob)

    def value_to_prob(self, value):
        if self.step is None:
            return self._numerical_to_prob(value)
        return self._to_prob_with_step(value)

    def get_config(self):
        config = super().get_config()
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
            step=proto.step or None,
            sampling=hp_utils.sampling_from_proto(proto.sampling),
            default=proto.default,
            conditions=conditions,
        )

    def to_proto(self):
        return keras_tuner_pb2.Float(
            name=self.name,
            min_value=self.min_value,
            max_value=self.max_value,
            step=self.step if self.step is not None else 0.0,
            sampling=hp_utils.sampling_to_proto(self.sampling),
            default=self.default,
            conditions=[c.to_proto() for c in self.conditions],
        )
