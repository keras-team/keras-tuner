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

import math

from keras_tuner.engine.hyperparameters import hp_utils
from keras_tuner.engine.hyperparameters import hyperparameter


class Numerical(hyperparameter.HyperParameter):
    """Super class for all numerical type hyperparameters."""

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
        super().__init__(name=name, default=default, **kwargs)
        self.max_value = max_value
        self.min_value = min_value
        self.step = step
        self.sampling = sampling
        self._check_sampling_arg()

    def _check_sampling_arg(self):
        if self.min_value > self.max_value:
            raise ValueError(
                f"For HyperParameters.{self.__class__.__name__}"
                f"(name='{self.name}'), "
                f"min_value {str(self.min_value)} is greater than "
                f"the max_value {str(self.max_value)}."
            )
        sampling_values = {"linear", "log", "reverse_log"}
        # This is for backward compatibility.
        # sampling=None was allowed and was the default value.
        if self.sampling is None:
            self.sampling = "linear"
        self.sampling = self.sampling.lower()
        if self.sampling not in sampling_values:
            raise ValueError(
                f"For HyperParameters.{self.__class__.__name__}"
                f"(name='{self.name}'), sampling must be one "
                f"of {sampling_values}"
            )

        if self.sampling in {"log", "reverse_log"} and self.min_value <= 0:
            raise ValueError(
                f"For HyperParameters.{self.__class__.__name__}"
                f"(name='{self.name}'), "
                f"sampling='{str(self.sampling)}' does not support "
                f"negative values, found min_value: {str(self.min_value)}."
            )
        if (
            self.sampling in {"log", "reverse_log"}
            and self.step is not None
            and self.step <= 1
        ):
            raise ValueError(
                f"For HyperParameters.{self.__class__.__name__}"
                f"(name='{self.name}'), "
                f"expected step > 1 with sampling='{str(self.sampling)}'. "
                f"Received: step={str(self.step)}."
            )

    def _sample_numerical_value(self, prob, max_value=None):
        """Sample a value with the cumulative prob in the given range."""
        if max_value is None:
            max_value = self.max_value
        if self.sampling == "linear":
            return prob * (max_value - self.min_value) + self.min_value
        elif self.sampling == "log":
            return self.min_value * math.pow(max_value / self.min_value, prob)
        elif self.sampling == "reverse_log":
            return (
                max_value
                + self.min_value
                - self.min_value * math.pow(max_value / self.min_value, 1 - prob)
            )

    def _numerical_to_prob(self, value, max_value=None):
        """Convert a numerical value to range [0.0, 1.0)."""
        if max_value is None:
            max_value = self.max_value
        if max_value == self.min_value:
            # Center the prob
            return 0.5
        if self.sampling == "linear":
            return (value - self.min_value) / (max_value - self.min_value)
        if self.sampling == "log":
            return math.log(value / self.min_value) / math.log(
                max_value / self.min_value
            )
        if self.sampling == "reverse_log":
            return 1.0 - math.log(
                (max_value + self.min_value - value) / self.min_value
            ) / math.log(max_value / self.min_value)

    def _get_n_values(self):
        """Get the total number of possible values using step."""
        if self.sampling == "linear":
            # +1 so that max_value may be sampled.
            return int((self.max_value - self.min_value) // self.step + 1)
        # For log and reverse_log
        # +1 so that max_value may be sampled.
        return int(math.log(self.max_value / self.min_value, self.step) + 1e-8) + 1

    def _get_value_by_index(self, index):
        """Get the index-th value in the range given step."""
        if self.sampling == "linear":
            return self.min_value + index * self.step
        if self.sampling == "log":
            return self.min_value * math.pow(self.step, index)
        if self.sampling == "reverse_log":
            return (
                self.max_value
                + self.min_value
                - self.min_value * math.pow(self.step, index)
            )

    def _sample_with_step(self, prob):
        """Sample a value with the cumulative prob in the given range.

        The range is divided evenly by `step`. So only sampling from a finite set of
        values. When calling the function, no need to use (max_value + 1) since the
        function takes care of the inclusion of max_value.
        """
        n_values = self._get_n_values()
        index = hp_utils.prob_to_index(prob, n_values)
        return self._get_value_by_index(index)

    @property
    def values(self):
        if self.step is None:
            # Evenly select 10 samples as the values.
            return tuple({self.prob_to_value(i * 0.1 + 0.05) for i in range(10)})
        n_values = self._get_n_values()
        return (self._get_value_by_index(i) for i in range(n_values))

    def _to_prob_with_step(self, value):
        """Convert to cumulative prob with step specified.

        When calling the function, no need to use (max_value + 1) since the function
        takes care of the inclusion of max_value.
        """
        if self.sampling == "linear":
            index = (value - self.min_value) // self.step
        if self.sampling == "log":
            index = math.log(value / self.min_value, self.step)
        if self.sampling == "reverse_log":
            index = math.log(
                (self.max_value - value + self.min_value) / self.min_value, self.step
            )
        n_values = self._get_n_values()
        return hp_utils.index_to_prob(index, n_values)
