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
import random

from ..engine import multi_execution_tuner
from ..engine import oracle as oracle_module
from ..engine import trial as trial_lib


class HyperbandOracle(oracle_module.Oracle):
    """Oracle class for Hyperband.

    Attributes:
        objective: String or `kerastuner.Objective`. If a string,
          the direction of the optimization (min or max) will be
          inferred.
        max_trials: Int. Total number of trials
            (model configurations) to test at most.
            Note that the oracle may interrupt the search
            before `max_trial` models have been tested.
        factor: Int. Reduction factor for the number of epochs
            and number of models for each bracket.
        min_epochs: Int. The minimum number of epochs to train a model.
        max_epochs: Int. The maximum number of epochs to train a model.
        seed: Int. Random seed.
        hyperparameters: HyperParameters class instance.
            Can be used to override (or register in advance)
            hyperparamters in the search space.
        tune_new_entries: Whether hyperparameter entries
            that are requested by the hypermodel
            but that were not specified in `hyperparameters`
            should be added to the search space, or not.
            If not, then the default value for these parameters
            will be used.
        allow_new_entries: Whether the hypermodel is allowed
            to request hyperparameter entries not listed in
            `hyperparameters`.
    """

    def __init__(self,
                 objective,
                 max_trials,
                 factor=3,
                 min_epochs=3,
                 max_epochs=10,
                 seed=None,
                 hyperparameters=None,
                 allow_new_entries=True,
                 tune_new_entries=True):
        super(HyperbandOracle, self).__init__(
            objective=objective,
            max_trials=max_trials,
            hyperparameters=hyperparameters,
            allow_new_entries=allow_new_entries,
            tune_new_entries=tune_new_entries)
        if min_epochs >= max_epochs:
            raise ValueError('max_epochs needs to be larger than min_epochs.')
        if factor < 2:
            raise ValueError('factor needs to be a int larger than 1.')
        self.seed = seed or random.randint(1, 1e4)
        self.factor = factor
        self.min_epochs = min_epochs
        self.max_epochs = max_epochs
        self._queue = []
        self._trial_count = 0
        self._running = {}
        self._trial_id_to_candidate_index = {}
        self._candidates = None
        self._candidate_score = None
        self._max_collisions = 20
        self._seed_state = self.seed
        self._tried_so_far = set()
        self._index_to_id = {}
        self._num_brackets = self._get_num_brackets()
        self._bracket_index = self._num_brackets
        self._model_sequence = self._get_model_sequence()
        self._epoch_sequence = self._get_epoch_sequence()

    def end_trial(self, trial_id, status):
        super(HyperbandOracle, self).end_trial(trial_id, status)
        self._running[trial_id] = False
        score = self.trials[trial_id].score
        self._candidate_score[
            self._trial_id_to_candidate_index[trial_id]] = score
        self.save()

    def _populate_space(self, trial_id):
        space = self.hyperparameters.space
        # Queue is not empty means it is in one bracket.
        if self._queue:
            return self._run_values(space, trial_id)

        # Wait the current bracket to finish.
        if any([value for key, value in self._running.items()]):
            return {'status': trial_lib.TrialStatus.IDLE}

        # Start the next bracket if not end of bandit.
        if self._bracket_index + 1 < self._num_brackets:
            self._bracket_index += 1
            self._select_candidates()
        # If the current band ends.
        else:
            self._bracket_index = 0
            self._generate_candidates()
            self._index_to_id = {}

        return self._run_values(space, trial_id)

    def _run_values(self, space, trial_id):
        self._trial_count += 1
        self._running[trial_id] = True
        candidate_index = self._queue.pop(0)
        if candidate_index not in self._index_to_id:
            self._index_to_id[candidate_index] = trial_id
        candidate = self._candidates[candidate_index]
        self._trial_id_to_candidate_index[trial_id] = candidate_index
        if candidate is not None:
            values = self._copy_values(space, candidate)
            if trial_id != self._index_to_id[candidate_index]:
                values['tuner/trial_id'] = self._index_to_id[candidate_index]
            values['tuner/epochs'] = self._epoch_sequence[self._bracket_index]
            return {'status': trial_lib.TrialStatus.RUNNING,
                    'values': values}
        return {'status': trial_lib.TrialStatus.STOPPED}

    @staticmethod
    def _copy_values(space, values):
        return_values = values.copy()
        for hyperparameter in space:
            if hyperparameter.name not in values:
                return_values[hyperparameter.name] = hyperparameter.default
        return return_values

    def _generate_candidates(self):
        self._candidates = []
        self._candidate_score = []
        num_models = self._model_sequence[0]

        for index in range(num_models):
            instance = self._new_trial()
            if instance is not None:
                self._candidates.append(instance)
                self._candidate_score.append(None)

        for index, instance in enumerate(self._candidates):
            self._queue.append(index)

    def _select_candidates(self):
        sorted_candidates = sorted(list(range(len(self._candidates))),
                                   key=lambda i: self._candidate_score[i])
        num_selected_candidates = self._model_sequence[self._bracket_index]
        for index in sorted_candidates[:num_selected_candidates]:
            self._queue.append(index)

    def _new_trial(self):
        """Fill a given hyperparameter space with values.

        Returns:
            A dictionary mapping parameter names to suggested values.
            Note that if the Oracle is keeping tracking of a large
            space, it may return values for more parameters
            than what was listed in `space`.
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
                    return None
                continue
            self._tried_so_far.add(values_hash)
            break
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

    def get_state(self):
        state = super(HyperbandOracle, self).get_state()
        state.update({
            'seed': self.seed,
            'factor': self.factor,
            'min_epochs': self.min_epochs,
            'max_epochs': self.max_epochs,
            'queue': copy.copy(self._queue),
            'trial_count': self._trial_count,
            'running': self._running,
            'trial_id_to_candidate_index': self._trial_id_to_candidate_index,
            'candidates': self._candidates,
            'candidate_score': self._candidate_score,
            'max_collisions': self._max_collisions,
            'seed_state': self._seed_state,
            'tried_so_far': list(self._tried_so_far),
            'index_to_id': self._index_to_id,
            'num_brackets': self._num_brackets,
            'bracket_index': self._bracket_index,
            'model_sequence': self._model_sequence,
            'epoch_sequence': self._epoch_sequence
        })
        return state

    def set_state(self, state):
        super(HyperbandOracle, self).set_state(state)
        self.seed = state['seed']
        self.factor = state['factor']
        self.min_epochs = state['min_epochs']
        self.max_epochs = state['max_epochs']
        self._queue = state['queue']
        self._trial_count = state['trial_count']
        self._running = state['running']
        self._trial_id_to_candidate_index = state['trial_id_to_candidate_index']
        self._candidates = state['candidates']
        self._candidate_score = state['candidate_score']
        self._max_collisions = state['max_collisions']
        self._seed_state = state['seed_state']
        self._tried_so_far = set(state['tried_so_far'])
        self._index_to_id = state['index_to_id']
        self._num_brackets = state['num_brackets']
        self._bracket_index = state['bracket_index']
        self._model_sequence = state['model_sequence']
        self._epoch_sequence = state['epoch_sequence']

        # Put the unfinished trials back into the queue to be trained again.
        for trial_id, value in self._running.items():
            if value:
                self._queue.append(self._trial_id_to_candidate_index[trial_id])
                self._running[trial_id] = False


class Hyperband(multi_execution_tuner.MultiExecutionTuner):
    """Variation of HyperBand algorithm.

    Reference:
        Li, Lisha, and Kevin Jamieson.
        ["Hyperband: A Novel Bandit-Based
         Approach to Hyperparameter Optimization."
        Journal of Machine Learning Research 18 (2018): 1-52](
            http://jmlr.org/papers/v18/16-558.html).


    Attributes:
        oracle: Instance of Oracle class.
        hypermodel: Instance of HyperModel class
            (or callable that takes hyperparameters
            and returns a Model instance).
        objective: String. Name of model metric to minimize
            or maximize, e.g. "val_accuracy".
        max_trials: Int. Total number of trials
            (model configurations) to test at most.
            Note that the oracle may interrupt the search
            before `max_trial` models have been tested.
        factor: Int. Reduction factor for the number of epochs
            and number of models for each bracket.
        min_epochs: Int. Minimum number of epochs to train a model.
        max_epochs: Int. Maximum number of epochs to train a model.
        seed: Int. Random seed.
        hyperparameters: HyperParameters class instance.
            Can be used to override (or register in advance)
            hyperparamters in the search space.
        tune_new_entries: Whether hyperparameter entries
            that are requested by the hypermodel
            but that were not specified in `hyperparameters`
            should be added to the search space, or not.
            If not, then the default value for these parameters
            will be used.
        allow_new_entries: Whether the hypermodel is allowed
            to request hyperparameter entries not listed in
            `hyperparameters`.
        **kwargs: Keyword arguments relevant to all `Tuner` subclasses.
            Please see the docstring for `Tuner`.
    """

    def __init__(self,
                 hypermodel,
                 objective,
                 max_trials,
                 factor=3,
                 min_epochs=3,
                 max_epochs=10,
                 seed=None,
                 hyperparameters=None,
                 tune_new_entries=True,
                 allow_new_entries=True,
                 **kwargs):
        oracle = HyperbandOracle(
            objective,
            max_trials,
            seed=seed,
            factor=factor,
            min_epochs=min_epochs,
            max_epochs=max_epochs,
            hyperparameters=hyperparameters,
            tune_new_entries=tune_new_entries,
            allow_new_entries=allow_new_entries)
        super(Hyperband, self).__init__(
            oracle=oracle,
            hypermodel=hypermodel,
            **kwargs)

    def run_trial(self, trial, *fit_args, **fit_kwargs):
        hp = trial.hyperparameters
        if 'tuner/epochs' in hp.values:
            fit_kwargs['epochs'] = hp.values['tuner/epochs']
        super(Hyperband, self).run_trial(trial, *fit_args, **fit_kwargs)

    def _build_model(self, hp):
        model = super(Hyperband, self)._build_model(hp)
        if 'tuner/trial_id' in hp.values:
            trial_id = hp.values['tuner/trial_id']
            history_trial = self.oracle.get_trial(trial_id)
            # Load best checkpoint from this trial.
            model.load_weights(self._get_checkpoint_fname(
                history_trial, history_trial.best_step))
        return model

    def _get_trial(self, trial_id):
        for temp_trial in self.trials:
            if temp_trial.trial_id == trial_id:
                return temp_trial
