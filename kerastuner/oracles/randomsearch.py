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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from ..engine import oracle as oracle_module
from ..engine import hyperparameters as hp_module

import random


class RandomSearch(oracle_module.Oracle):

    def __init__(self, seed=None):
        super(RandomSearch, self).__init__()
        self.seed = seed or random.randint(1, 1e4)
        # Incremented at every call to `populate_space`.
        self._seed_state = self.seed
        # Hashes of values tried so far.
        self._tried_so_far = set()
        # Maximum number of identical values that can be generated
        # before we consider the space to be exhausted.
        self._max_collisions = 5

    def populate_space(self, trial_id, space):
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
        self.update_space(space)
        collisions = 0
        while 1:
            # Generate a set of random values.
            values = {}
            for p in space:
                values[p.name] = p.random_sample(self._seed_state)
                self._seed_state += 1
            # Keep trying until the set of values is unique,
            # or until we exit due to too many collisions.
            values_hash = self._compute_values_hash(values)
            if values_hash in self._tried_so_far:
                collisions += 1
                if collisions > self._max_collisions:
                    return {'status': 'EXIT'}
                continue
            self._tried_so_far.add(values_hash)
            break
        return {'values': values, 'status': 'RUN'}
