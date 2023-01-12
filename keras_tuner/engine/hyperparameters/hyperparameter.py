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

import random

from keras_tuner import utils
from keras_tuner.engine import conditions as conditions_mod


class HyperParameter:
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

        conditions = utils.to_list(conditions) if conditions else []
        self.conditions = conditions

    def get_config(self):
        conditions = [conditions_mod.serialize(c) for c in self.conditions]
        return {"name": self.name, "default": self.default, "conditions": conditions}

    @property
    def default(self):
        return self._default

    @property
    def values(self):
        """Return a iterable of all possible values of the hp."""
        raise NotImplementedError

    def random_sample(self, seed=None):
        random_state = random.Random(seed)
        prob = float(random_state.random())
        return self.prob_to_value(prob)

    def prob_to_value(self, prob):
        """Convert cumulative probability in range [0.0, 1.0) to hp value."""
        raise NotImplementedError

    def value_to_prob(self, value):
        """Convert a hp value to cumulative probability in range [0.0, 1.0)."""
        raise NotImplementedError

    @classmethod
    def from_config(cls, config):
        config["conditions"] = [
            conditions_mod.deserialize(c) for c in config["conditions"]
        ]
        return cls(**config)
