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
import math
import random

from ..engine import multi_execution_tuner
from ..engine import oracle as oracle_module


class HyperbandOracle(oracle_module.Oracle):
    """Oracle class for Hyperband.

    Note that to use this Oracle with your own subclassed Tuner, your Tuner
    class must be able to handle in `Tuner.run_trial` three special hyperparameters
    that will be set by this Tuner:

      - "tuner/trial_id": String, optionally set. The trial_id of the Trial to load
          from when starting this trial.
      - "tuner/initial_epoch": Int, always set. The initial epoch the Trial should be
          started from.
      - "tuner/epochs": Int, always set. The cumulative number of epochs this Trial
          should be trained.

    These hyperparameters will be set during the "successive halving" portion
    of the Hyperband algorithm.

    Example `run_trial`:

    ```
    def run_trial(self, trial, *args, **kwargs):
        hp = trial.hyperparameters
        if "tuner/trial_id" in hp:
            past_trial = self.oracle.get_trial(hp['tuner/trial_id'])
            model = self.load_model(past_trial)
        else:
            model = self.hypermodel.build(hp)

        initial_epoch = hp['tuner/initial_epoch']
        last_epoch = hp['tuner/epochs']

        for epoch in range(initial_epoch, last_epoch):
            self.on_epoch_begin(...)
            for step in range(...):
                # Run model training step here.
            self.on_epoch_end(...)
    ```

    # Arguments:
        objective: String or `kerastuner.Objective`. If a string,
          the direction of the optimization (min or max) will be
          inferred.
        max_epochs: Int. The maximum number of epochs to train one model. It is
          recommended to set this to a value slightly higher than the expected epochs
          to convergence for your largest Model, and to use early stopping during
          training (for example, via `tf.keras.callbacks.EarlyStopping`).
        factor: Int. Reduction factor for the number of epochs
            and number of models for each bracket.
        hyperband_iterations: Int >= 1. The number of times to iterate over the full
          Hyperband algorithm. One iteration will run approximately
          `max_epochs * (math.log(max_epochs, factor) ** 2)` cumulative epochs
          across all trials. It is recommended to set this to as high a value
          as is within your resource budget.
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
                 max_epochs,
                 factor=3,
                 hyperband_iterations=1,
                 seed=None,
                 hyperparameters=None,
                 allow_new_entries=True,
                 tune_new_entries=True):
        super(HyperbandOracle, self).__init__(
            objective=objective,
            hyperparameters=hyperparameters,
            allow_new_entries=allow_new_entries,
            tune_new_entries=tune_new_entries,
            seed=seed)
        if factor < 2:
            raise ValueError('factor needs to be a int larger than 1.')

        self.hyperband_iterations = hyperband_iterations or float('inf')
        self.max_epochs = max_epochs
        # Minimum epochs before successive halving, Hyperband sweeps through varying
        # degress of aggressiveness.
        self.min_epochs = 1
        self.factor = factor

        self.seed = seed or random.randint(1, 10000)
        self._max_collisions = 20
        self._seed_state = self.seed
        self._tried_so_far = set()

        self._current_iteration = 0
        # Start with most aggressively halving bracket.
        self._current_bracket = self._get_num_brackets() - 1
        self._brackets = []

        self._start_new_bracket()

    def _populate_space(self, trial_id):
        self._remove_completed_brackets()

        for bracket in self._brackets:
            bracket_num = bracket['bracket_num']
            rounds = bracket['rounds']

            if len(rounds[0]) < self._get_size(bracket_num, round_num=0):
                # Populate the initial random trials for this bracket.
                return self._random_trial(trial_id, bracket)
            else:
                # Try to populate incomplete rounds for this bracket.
                for round_num in range(1, len(rounds)):
                    round_info = rounds[round_num]
                    past_round_info = rounds[round_num - 1]
                    size = self._get_size(bracket_num, round_num)
                    past_size = self._get_size(bracket_num, round_num - 1)

                    # If more trials from the last round are ready than will be
                    # thrown out, we can select the best to run for the next round.
                    already_selected = [info['past_id'] for info in round_info]
                    candidates = [self.trials[info['id']]
                                  for info in past_round_info
                                  if info['id'] not in already_selected]
                    candidates = [t for t in candidates if t.status == 'COMPLETED']
                    if len(candidates) > past_size - size:
                        sorted_candidates = sorted(
                            candidates,
                            key=lambda t: t.score,
                            reverse=self.objective.direction == 'max')
                        best_trial = sorted_candidates[0]

                        values = best_trial.hyperparameters.values.copy()
                        values['tuner/trial_id'] = best_trial.trial_id
                        values['tuner/epochs'] = self._get_epochs(
                            bracket_num, round_num)
                        values['tuner/initial_epoch'] = self._get_epochs(
                            bracket_num, round_num - 1)
                        values['tuner/bracket'] = self._current_bracket
                        values['tuner/round'] = round_num

                        round_info.append({'past_id': best_trial.trial_id,
                                           'id': trial_id})
                        return {'status': 'RUNNING', 'values': values}

        # This is reached if no trials from current brackets can be run.

        # Max sweeps has been reached, no more brackets should be created.
        if (self._current_bracket == 0 and
                self._current_iteration + 1 >= self.hyperband_iterations):
            # Stop creating new brackets, but wait to complete other brackets.
            if self.ongoing_trials:
                return {'status': 'IDLE'}
            else:
                self._increment_bracket_num()
                return {'status': 'STOPPED'}
        # Create a new bracket.
        else:
            self._increment_bracket_num()
            self._start_new_bracket()
            return self._random_trial(trial_id, self._brackets[-1])

    def _start_new_bracket(self):
        rounds = []
        for _ in range(self._get_num_rounds(self._current_bracket)):
            rounds.append([])
        bracket = {'bracket_num': self._current_bracket, 'rounds': rounds}
        self._brackets.append(bracket)

    def _increment_bracket_num(self):
        self._current_bracket -= 1
        if self._current_bracket < 0:
            self._current_bracket = self._get_num_brackets() - 1
            self._current_iteration += 1
            if self._current_iteration > self.hyperband_iterations:
                self._current_bracket = 0

    def _remove_completed_brackets(self):
        # Filter out completed brackets.
        def _bracket_is_incomplete(bracket):
            bracket_num = bracket['bracket_num']
            rounds = bracket['rounds']
            last_round = len(rounds) - 1
            if len(rounds[last_round]) == self._get_size(bracket_num, last_round):
                # All trials have been created for the current bracket.
                return False
            return True
        self._brackets = list(filter(_bracket_is_incomplete, self._brackets))

    def _random_trial(self, trial_id, bracket):
        bracket_num = bracket['bracket_num']
        rounds = bracket['rounds']
        values = self._random_values()
        if values:
            values['tuner/epochs'] = self._get_epochs(bracket_num, 0)
            values['tuner/initial_epoch'] = 0
            values['tuner/bracket'] = self._current_bracket
            values['tuner/round'] = 0
            rounds[0].append({'past_id': None, 'id': trial_id})
            return {'status': 'RUNNING', 'values': values}
        elif self.ongoing_trials:
            # Can't create new random values, but successive halvings may still
            # be needed.
            return {'status': 'IDLE'}
        else:
            # Collision and no ongoing trials should trigger an exit.
            return {'status': 'STOPPED'}

    def _get_size(self, bracket_num, round_num):
        # Set up so that each bracket takes approx. the same amount of resources.
        bracket0_end_size = math.ceil(1 + math.log(self.max_epochs, self.factor))
        bracket_end_size = bracket0_end_size / (bracket_num + 1)
        return math.ceil(bracket_end_size * self.factor**(bracket_num - round_num))

    def _get_epochs(self, bracket_num, round_num):
        return math.ceil(self.max_epochs / self.factor**(bracket_num - round_num))

    def _get_num_rounds(self, bracket_num):
        # Bracket 0 just runs random search, others do successive halving.
        return bracket_num + 1

    def _get_num_brackets(self):
        epochs = self.max_epochs
        brackets = 0
        while epochs >= self.min_epochs:
            epochs = epochs / self.factor
            brackets += 1
        return brackets

    def get_state(self):
        state = super(HyperbandOracle, self).get_state()
        state.update({
            'hyperband_iterations': self.hyperband_iterations,
            'max_epochs': self.max_epochs,
            'min_epochs': self.min_epochs,
            'factor': self.factor,
            'brackets': self._brackets,
            'current_bracket': self._current_bracket,
            'current_iteration': self._current_iteration
        })
        return state

    def set_state(self, state):
        super(HyperbandOracle, self).set_state(state)
        self.hyperband_iterations = state['hyperband_iterations']
        self.max_epochs = state['max_epochs']
        self.min_epochs = state['min_epochs']
        self.factor = state['factor']
        self._brackets = state['brackets']
        self._current_bracket = state['current_bracket']
        self._current_iteration = state['current_iteration']


class Hyperband(multi_execution_tuner.MultiExecutionTuner):
    """Variation of HyperBand algorithm.

    Reference:
        Li, Lisha, and Kevin Jamieson.
        ["Hyperband: A Novel Bandit-Based
         Approach to Hyperparameter Optimization."
        Journal of Machine Learning Research 18 (2018): 1-52](
            http://jmlr.org/papers/v18/16-558.html).


    # Arguments
        hypermodel: Instance of HyperModel class
            (or callable that takes hyperparameters
            and returns a Model instance).
        objective: String. Name of model metric to minimize
            or maximize, e.g. "val_accuracy".
        max_epochs: Int. The maximum number of epochs to train one model. It is
          recommended to set this to a value slightly higher than the expected time
          to convergence for your largest Model, and to use early stopping during
          training (for example, via `tf.keras.callbacks.EarlyStopping`).
        factor: Int. Reduction factor for the number of epochs
            and number of models for each bracket.
        hyperband_iterations: Int >= 1. The number of times to iterate over the full
          Hyperband algorithm. One iteration will run approximately
          `max_epochs * (math.log(max_epochs, factor) ** 2)` cumulative epochs
          across all trials. It is recommended to set this to as high a value
          as is within your resource budget.
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
                 max_epochs,
                 factor=3,
                 hyperband_iterations=1,
                 seed=None,
                 hyperparameters=None,
                 tune_new_entries=True,
                 allow_new_entries=True,
                 **kwargs):
        oracle = HyperbandOracle(
            objective,
            max_epochs=max_epochs,
            factor=factor,
            hyperband_iterations=hyperband_iterations,
            seed=seed,
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
            fit_kwargs['initial_epoch'] = hp.values['tuner/initial_epoch']
        super(Hyperband, self).run_trial(trial, *fit_args, **fit_kwargs)

    def _build_model(self, hp):
        model = super(Hyperband, self)._build_model(hp)
        if 'tuner/trial_id' in hp.values:
            trial_id = hp.values['tuner/trial_id']
            history_trial = self.oracle.get_trial(trial_id)
            # Load best checkpoint from this trial.
            model.load_weights(self._get_checkpoint_fname(
                history_trial.trial_id, history_trial.best_step))
        return model
