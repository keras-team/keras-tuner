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


import copy
import queue
import random
from ..engine import tuner as tuner_module
from ..engine import execution as execution_module
from ..engine import tuner_utils
from ..engine import oracle as oracle_module


class UltraBandOracle(oracle_module.Oracle):
    """Oracle class for UltraBand.

        Args:
        max_trials: Int. Total number of trials
            (model configurations) to test at most.
            Note that the oracle may interrupt the search
            before `max_trial` models have been tested.
        seed: Int. The random seed. If None, it would use a random number.
        factor: Int. The factor of the change of number of epochs and the number of models per bracket.
        min_epochs: Int. The minimum number of epochs to train a model.
        max_epochs: Int. The maximum number of epochs to train a model.
    """

    def __init__(self,
                 max_trials=200,
                 seed=None,
                 factor=3,
                 min_epochs=3,
                 max_epochs=10):
        super().__init__()
        if min_epochs >= max_epochs:
            raise ValueError('max_epochs needs to be larger than min_epochs.')
        if factor < 2:
            raise ValueError('factor needs to be a int larger than 1.')
        self.trials = max_trials
        self.seed = seed or random.randint(1, 1e4)
        self.factor = factor
        self.min_epochs = min_epochs
        self.max_epochs = max_epochs
        self._queue = queue.Queue()
        self._bracket_index = 0
        self._trials_count = 0
        self._running = {}
        self._trial_id_to_candidate_index = {}
        self._candidates = None
        self._candidate_score = None
        self._max_collisions = 20
        self._seed_state = self.seed
        self._tried_so_far = set()
        self._index_to_id = {}
        self._num_brackets = self._get_num_brackets()
        self._model_sequence = self._get_model_sequence()
        self._epoch_sequence = self._get_epoch_sequence()

    def result(self, trial_id, score):
        self._running[trial_id] = False
        self._candidate_score[self._trial_id_to_candidate_index[trial_id]] = score

    def populate_space(self, trial_id, space):
        if (self._trials_count >= self.trials and
                not any([value for key, value in self._running.items()])):
            return {'status': 'EXIT'}
        if self._trials_count == 0:
            self._trials_count += 1
            return {'status': 'RUN', 'values': self._copy_values(space, {})}

        # queue not empty means it is in one bracket
        if not self._queue.empty():
            return self._run_values(space, trial_id)

        # check if the current batch ends
        if (self._bracket_index >= self._num_brackets and
                not any([value for key, value in self._running.items()])):
            self._bracket_index = 0

        # check if the band ends
        if self._bracket_index == 0:
            # band ends
            self._generate_candidates(space)
            self._index_to_id = {}
        else:
            # bracket ends
            if any([value for key, value in self._running.items()]):
                return {'status': 'IDLE'}
            self._select_candidates()
        self._bracket_index += 1

        return self._run_values(space, trial_id)

    def _run_values(self, space, trial_id):
        self._trials_count += 1
        self._running[trial_id] = True
        candidate_index = self._queue.get()
        if candidate_index not in self._index_to_id:
            self._index_to_id[candidate_index] = trial_id
        candidate = self._candidates[candidate_index]
        self._trial_id_to_candidate_index[trial_id] = candidate_index
        if candidate is not None:
            values = self._copy_values(space, candidate)
            values['tuner/trial_id'] = self._index_to_id[candidate_index]
            return {'status': 'RUN', 'values': values}
        return {'status': 'EXIT'}

    @staticmethod
    def _copy_values(space, values):
        return_values = {}
        for hyperparameter in space:
            if hyperparameter.name in values:
                return_values[hyperparameter.name] = values[hyperparameter.name]
            else:
                return_values[hyperparameter.name] = hyperparameter.default
        return return_values

    def _generate_candidates(self, space):
        self._candidates = []
        self._candidate_score = []
        num_models = self._model_sequence[0]

        for index in range(num_models):
            instance = self._new_trial(space)
            if instance is not None:
                self._candidates.append(instance)
                self._candidate_score.append(None)

        for index, instance in enumerate(self._candidates):
            self._queue.put(index)

    def _select_candidates(self):
        for index in sorted(
                list(range(len(self._candidates))),
                key=lambda i: self._candidate_score[i],
                reverse=True,
        )[:self._model_sequence[self._bracket_index]]:
            self._queue.put(index)

    def _new_trial(self, space):
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
                    return None
                continue
            self._tried_so_far.add(values_hash)
            break
        values['tuner/epochs'] = self._epoch_sequence[self._bracket_index]
        return values

    def _get_num_brackets(self):
        """Compute the number of brackets based on the scaling factor"""
        n = 1
        v = self.min_epochs
        while v < self.max_epochs:
            v *= self.factor
            n += 1
        return n

    def _get_model_sequence(self):
        sizes = []
        size = self.min_epochs
        for _ in range(self._num_brackets - 1):
            sizes.append(int(size))
            size *= self.factor
        sizes.append(self.max_epochs)
        sizes.reverse()
        return sizes

    def _get_epoch_sequence(self):
        """Compute the sequence of epochs per bracket."""
        sizes = []
        size = self.min_epochs
        for _ in range(self._num_brackets - 1):
            sizes.append(int(size))
            size *= self.factor
        sizes.append(self.max_epochs)

        previous_size = 0
        output_sizes = []
        for size in sizes:
            output_sizes.append(size - previous_size)
            previous_size = size
        return output_sizes


class UltraBand(tuner_module.Tuner):
    """Variation of HyperBand algorithm.

    Args:
        oracle: Instance of Oracle class.
        hypermodel: Instnace of HyperModel class
            (or callable that takes hyperparameters
            and returns a Model isntance).
        objective: String. Name of model metric to minimize
            or maximize, e.g. "val_accuracy".
        max_trials: Int. Total number of trials
            (model configurations) to test at most.
            Note that the oracle may interrupt the search
            before `max_trial` models have been tested.
        seed: Int. The random seed. If None, it would use a random number.
        factor: Int. The factor of the change of number of epochs and the number of models per bracket.
        min_epochs: Int. The minimum number of epochs to train a model.
        max_epochs: Int. The maximum number of epochs to train a model.
    """

    def __init__(self,
                 hypermodel,
                 objective,
                 max_trials,
                 seed=None,
                 factor=3,
                 min_epochs=3,
                 max_epochs=10,
                 **kwargs):
        oracle = UltraBandOracle(max_trials,
                                 seed,
                                 factor,
                                 min_epochs,
                                 max_epochs)
        super(UltraBand, self).__init__(
            oracle,
            hypermodel,
            objective,
            max_trials,
            **kwargs)

    def on_execution_begin(self, trial, execution, model):
        super(UltraBand, self).on_execution_begin(trial, execution, model)
        hp = trial.hyperparameters
        if 'tuner/trial_id' in hp.values:
            history_trial = self._get_trial(hp.values['tuner/trial_id'])
            if history_trial.executions[0].best_checkpoint is not None:
                best_checkpoint = history_trial.executions[0].best_checkpoint + '-weights.h5'
                model.load_weights(best_checkpoint)

    def run_trial(self, trial, hp, fit_args, fit_kwargs):
        if 'tuner/epochs' in hp.values:
            fit_kwargs['epochs'] = hp.values['tuner/epochs']
        super(UltraBand, self).run_trial(trial, hp, fit_args, fit_kwargs)

    def _get_trial(self, trial_id):
        for temp_trial in self.trials:
            if temp_trial.trial_id == trial_id:
                return temp_trial

    @classmethod
    def load(cls, filename):
        # TODO
        raise NotImplementedError

    def save(self):
        # TODO
        raise NotImplementedError

    def report_status(self, trial_id, status):
        # TODO
        raise NotImplementedError
