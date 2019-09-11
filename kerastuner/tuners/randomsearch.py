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

"Basic random search tuner."

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from ..engine import tuner as tuner_module
from ..engine import oracle as oracle_module
from ..engine import hyperparameters as hp_module
from ..abstractions.tensorflow import TENSORFLOW_UTILS as tf_utils

import random
import json


class RandomSearchOracle(oracle_module.Oracle):

    def __init__(self, seed=None):
        super(RandomSearchOracle, self).__init__()
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

    def save(self, fname):
        state = {
            'seed': self.seed,
            'seed_state': self._seed_state,
            'tried_so_far': list(self._tried_so_far),
        }
        state_json = json.dumps(state)
        tf_utils.write_file(fname, state_json)

    def reload(self, fname):
        state_data = tf_utils.read_file(fname)
        state = json.loads(state_data)
        self.seed = state['seed']
        self._seed_state = state['seed_state']
        self._tried_so_far = set(state['tried_so_far'])


class RandomSearch(tuner_module.Tuner):
    """Random search tuner.

    Args:
        hypermodel: Instance of HyperModel class
            (or callable that takes hyperparameters
            and returns a Model isntance).
        objective: String. Name of model metric to minimize
            or maximize, e.g. "val_accuracy".
        max_trials: Int. Total number of trials
            (model configurations) to test at most.
            Note that the oracle may interrupt the search
            before `max_trial` models have been tested.
        seed: Int. Random seed.
    """

    def __init__(self,
                 hypermodel,
                 objective,
                 max_trials,
                 seed=None,
                 **kwargs):
        self.seed = seed
        oracle = RandomSearchOracle(seed)
        super(RandomSearch, self).__init__(
            oracle,
            hypermodel,
            objective,
            max_trials,
            **kwargs)
