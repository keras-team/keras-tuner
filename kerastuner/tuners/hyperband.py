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
        self._max_collisions = 20
        self._seed_state = self.seed
        self._tried_so_far = set()

        self._brackets = [[]]*self._num_brackets

    def _populate_space(self, trial_id):
        # All trials have been created for the current band, start a new band.
        if len(self._brackets[-1]) == self._bracket_sizes[-1]:
            self._brackets = [[]]*self._num_brackets

        # Populate the initial bracket for this band.
        if len(self._brackets[0]) < self._bracket_sizes[0]:
            values = self._random_trial()
            if values:
                self._brackets[0].append({'past_id': None, 'id': trial_id})
                return {'status': trial_lib.TrialStatus.RUNNING,
                        'values': values}
            else:
                return {'status': trial_lib.TrialStatus.STOPPED}

        # Populate downstream brackets for this band.
        for i, bracket in enumerate(self._brackets[1:]):
            past_bracket = self._brackets[i - 1]
            size = self._bracket_sizes[i]
            past_size = self._bracket_sizes[i - 1]

            if len(bracket) == size:
                continue

            already_running = [ele['past_id'] for ele in bracket]
            candidates = []
            for ele in past_bracket:
                trial_id = ele['id']
                if trial_id not in already_running:
                    trial = self.trials[trial_id]
                    if trial.status == 'COMPLETED':
                        candidates.append(trial)

            # Enough trials from last bracket have completed to select a trial
            # for the next bracket (there are more models to run than the number
            # to throw out for next bracket).
            if len(candidates) > past_size - size:
                sorted_candidates = sorted(candidates,
                                           key=lambda t: t.score,
                                           reverse=self.objective.direction == 'max')
                best_trial = sorted_candidates[0]

                values = best_trial.hyperparameters.values.copy()
                values['tuner/trial_id'] = best_trial.trial_id
                values['tuner/epochs'] = self._bracket_epochs[i]
                values['tuner/initial_epoch'] = self._bracket_epochs[i - 1]

                bracket.append({'past_id': best_trial.trial_id, 'id': trial_id})

                return {'status': trial_lib.TrialStatus.RUNNING,
                        'values': values}

        # No trials from any bracket can be run until more trials are completed.
        return {'status': trial_lib.TrialStatus.IDLE}

    def _random_trial(self):
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

    @property
    def _num_brackets(self):
        n = 1
        v = self.min_epochs
        while v < self.max_epochs:
            v *= self.factor
            n += 1
        return n

    @property
    def _bracket_sizes(self):
        sizes = []
        size = self.min_epochs
        for _ in range(self._num_brackets - 1):
            sizes.append(int(size))
            size *= self.factor
        sizes.append(self.max_epochs)
        sizes.reverse()
        return sizes

    @property
    def _bracket_epochs(self):
        """Compute the sequence of epochs per bracket."""
        sizes = []
        size = self.min_epochs
        for _ in range(self._num_brackets - 1):
            sizes.append(int(size))
            size *= self.factor
        sizes.append(self.max_epochs)
        return sizes

    def get_state(self):
        state = super(HyperbandOracle, self).get_state()
        state.update({
            'seed': self.seed,
            'factor': self.factor,
            'min_epochs': self.min_epochs,
            'max_epochs': self.max_epochs,
            'max_collisions': self._max_collisions,
            'seed_state': self._seed_state,
            'tried_so_far': list(self._tried_so_far),
            'brackets': self._brackets
        })
        return state

    def set_state(self, state):
        super(HyperbandOracle, self).set_state(state)
        self.seed = state['seed']
        self.factor = state['factor']
        self.min_epochs = state['min_epochs']
        self.max_epochs = state['max_epochs']
        self._max_collisions = state['max_collisions']
        self._seed_state = state['seed_state']
        self._tried_so_far = set(state['tried_so_far'])
        self._brackets = state['brackets']


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
            fit_kwargs['initial_epoch'] = hp.values['tuner/epochs']
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
