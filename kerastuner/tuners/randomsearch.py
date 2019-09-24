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

from ..engine import trial as trial_lib
from ..engine import tuner as tuner_module
from ..engine import oracle as oracle_module
from ..engine import hyperparameters as hp_module
from ..abstractions.tensorflow import TENSORFLOW_UTILS as tf_utils

import random
import json


class RandomSearchOracle(oracle_module.Oracle):

    def __init__(self,
                 objective,
                 max_trials,
                 seed=None,
                 **kwargs):
        super(RandomSearchOracle, self).__init__(
            objective=objective, max_trials=max_trials, **kwargs)
        self.seed = seed or random.randint(1, 1e4)
        # Incremented at every call to `populate_space`.
        self._seed_state = self.seed
        # Hashes of values tried so far.
        self._tried_so_far = set()
        # Maximum number of identical values that can be generated
        # before we consider the space to be exhausted.
        self._max_collisions = 5

    def _populate_space(self, _):
        """Fill the hyperparameter space with values.

        Args:
          `trial_id`: The id for this Trial.

        Returns:
            A dictionary with keys "values" and "status", where "values" is
            a mapping of parameter names to suggested values, and "status"
            is the TrialStatus that should be returned for this trial (one
            of "RUNNING", "IDLE", or "STOPPED").
        """
        collisions = 0
        while 1:
            # Generate a set of random values.
            values = {}
            for p in self.hyperparameters.space:
                values[p.name] = p.random_sample(self._seed_state)
                self._seed_state += 1
            # Keep trying until the set of values is unique,
            # or until we exit due to too many collisions.
            values_hash = self._compute_values_hash(values)
            if values_hash in self._tried_so_far:
                collisions += 1
                if collisions > self._max_collisions:
                    return {'status': trial_lib.TrialStatus.STOPPED,
                            'values': None}
                continue
            self._tried_so_far.add(values_hash)
            break
        return {'status': trial_lib.TrialStatus.RUNNING,
                'values': values}

    def get_state(self):
        state = super(RandomSearchOracle, self).get_state()
        state.update({
            'seed': self.seed,
            'seed_state': self._seed_state,
            'tried_so_far': list(self._tried_so_far),
        })
        return state

    def set_state(self, state):
        super(RandomSearchOracle, self).set_state(state)
        self.seed = state['seed']
        self._seed_state = state['seed_state']
        self._tried_so_far = set(state['tried_so_far'])


class RandomSearch(tuner_module.Tuner):
    """Random search tuner.

    Args:
        hypermodel: Instance of HyperModel class
            (or callable that takes hyperparameters
            and returns a Model instance).
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
        oracle = RandomSearchOracle(
            objective=objective,
            max_trials=max_trials,
            seed=seed,
            hyperparameters=kwargs.pop('hyperparameters', None),
            tune_new_entries=kwargs.pop('tune_new_entries', True),
            allow_new_entries=kwargs.pop('allow_new_entries', True),
            executions_per_trial=kwargs.pop('executions_per_trial', 1))
        super(RandomSearch, self).__init__(
            oracle,
            hypermodel,
            **kwargs)
