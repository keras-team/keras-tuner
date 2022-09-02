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

    def _sample_with_step(self, prob):
        """Sample a value with the cumulative prob in the given range.

        The range is divided evenly by `step`. So only sampling from a finite set of
        values. When calling the function, no need to use (max_value + 1) since the
        function takes care of the inclusion of max_value.
        """
        if self.sampling == "linear":
            # +1 so that max_value may be sampled.
            n_values = (self.max_value - self.min_value) // self.step + 1
            index = hp_utils.prob_to_index(prob, n_values)
            return self.min_value + index * self.step
        if self.sampling == "log":
            # +1 so that max_value may be sampled.
            n_values = int(math.log(self.max_value / self.min_value, self.step)) + 1
            index = hp_utils.prob_to_index(prob, n_values)
            return self.min_value * math.pow(self.step, index)
        if self.sampling == "reverse_log":
            # +1 so that max_value may be sampled.
            n_values = int(math.log(self.max_value / self.min_value, self.step)) + 1
            index = hp_utils.prob_to_index(prob, n_values)
            return (
                self.max_value
                + self.min_value
                - self.min_value * math.pow(self.step, index)
            )

    def _to_prob_with_step(self, value):
        """Convert to cumulative prob with step specified.

        When calling the function, no need to use (max_value + 1) since the function
        takes care of the inclusion of max_value.
        """
        if self.sampling == "linear":
            index = (value - self.min_value) // self.step
            # +1 so that max_value may be sampled.
            n_values = (self.max_value - self.min_value) // self.step + 1
            return hp_utils.index_to_prob(index, n_values)
        if self.sampling == "log":
            index = math.log(value / self.min_value, self.step)
            # +1 so that max_value may be sampled.
            n_values = int(math.log(self.max_value / self.min_value, self.step)) + 1
            return hp_utils.index_to_prob(index, n_values)
        if self.sampling == "reverse_log":
            index = math.log(
                (self.max_value - value + self.min_value) / self.min_value, self.step
            )
            # +1 so that max_value may be sampled.
            n_values = int(math.log(self.max_value / self.min_value, self.step)) + 1
            return hp_utils.index_to_prob(index, n_values)
