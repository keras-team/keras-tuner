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
"Random search Oracle class."

from ..engine import oracle as oracle_module
from ..engine import hyperparameters as hp_module


class RandomSearch(oracle_module.Oracle):

    def __init__(self, seed):
        self.seed = seed
        self._seed_state = seed

    def populate_space(self, space):
        """Fill a given hyperparameter space with values.

        Args:
            space: A list of HyperParameter objects
                to provide values for.

        Returns:
            A dictionary mapping parameter names to suggested values.
            Note that if the Oracle is keeping tracking of a large
            space, it may return values for more parameters
            than what was listed in `space`.
        """
        self._raise_if_unknown_hyperparameter(space)
        values = {}
        for p in space:
            values[p.name] = p.random_sample(self._seed_state)
            self._seed_state += 1
        return values
