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
"Oracle base class."

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import enum
import hashlib
import json

from tensorflow import keras
from kerastuner.engine import hyperparameters as hp_module
from kerastuner.engine import metrics_tracking
from kerastuner.engine import stateful
from kerastuner.engine import trial as trial_lib
from ..abstractions.tensorflow import TENSORFLOW_UTILS as tf_utils


Objective = collections.namedtuple('Objective', 'name direction')


class Oracle(stateful.Stateful):

    def __init__(self,
                 objective,
                 max_trials=None,
                 hyperparameters=None,
                 allow_new_entries=True,
                 tune_new_entries=True,
                 executions_per_trial=1):
        self.objective = _format_objective(objective)
        if not hyperparameters:
            if not tune_new_entries:
                raise ValueError(
                    'If you set `tune_new_entries=False`, you must'
                    'specify the search space via the '
                    '`hyperparameters` argument.')
            if not allow_new_entries:
                raise ValueError(
                    'If you set `allow_new_entries=False`, you must'
                    'specify the search space via the '
                    '`hyperparameters` argument.')
            self.hyperparameters = hp_module.HyperParameters()
        else:
            self.hyperparameters = hyperparameters
        self.allow_new_entries = allow_new_entries
        self.tune_new_entries = tune_new_entries

        # Each hyperparameter combination will be repeated this number of times.
        # Scores of each execution will be averaged.
        self.executions_per_trial = executions_per_trial
        self.max_trials = max_trials * executions_per_trial
        self._last_trial_execution = 0
        self._last_hps = None

        # trial_id -> Trial
        self.trials = {}
        # tuner_id -> Trial
        self.ongoing_trials = {}

    def get_space(self):
        return self.hyperparameters.copy()

    def update_space(self, hyperparameters):
        """Add new hyperparameters to the tracking space.

        Already recorded parameters get ignored.

        Args:
            hyperparameters: An updated HyperParameters object.
        """
        ref_names = {hp.name for hp in self.hyperparameters.space}
        new_hps = [hp for hp in hyperparameters.space
                   if hp.name not in ref_names]

        if new_hps and not self.allow_new_entries:
            raise RuntimeError('`allow_new_entries` is `False`, but found '
                               'new entries {}'.format(new_hps))

        if not self.tune_new_entries:
            # New entries should always use the default value.
            return

        for hp in new_hps:
            self.hyperparameters.register(
                hp.name, hp.__class__.__name__, hp.get_config())

    def _populate_space(self, trial_id):
        """Fill the hyperparameter space with values for a trial.

        Args:
          `trial_id`: The id for this Trial.

        Returns:
            A dictionary with keys "values" and "status", where "values" is
            a mapping of parameter names to suggested values, and "status"
            is the TrialStatus that should be returned for this trial (one
            of "RUNNING", "IDLE", or "STOPPED").
        """
        raise NotImplementedError

    def create_trial(self, tuner_id):
        # Allow for multi-worker DistributionStrategy within a Trial.
        if tuner_id in self.ongoing_trials:
            return self.ongoing_trials[tuner_id]

        trial_id = trial_lib.generate_trial_id()

        if self._last_trial_execution > 0:
            # Repeat the current hyperparameter combination.
            status = trial_lib.TrialStatus.RUNNING
            values = self._last_hps
        elif self.max_trials and len(self.trials.items()) >= self.max_trials:
            status = trial_lib.TrialStatus.STOPPED
            values = None
        else:
            response = self._populate_space(trial_id)
            status = response['status']
            values = response['values'] if 'values' in response else None

        self._last_hps = values
        self._last_trial_execution += 1
        if self._last_trial_execution == self.executions_per_trial:
            self._last_trial_execution = 0

        hyperparameters = self.hyperparameters.copy()
        hyperparameters.values = values
        trial = trial_lib.Trial(
            hyperparameters=hyperparameters,
            trial_id=trial_id,
            status=status)

        if status == trial_lib.TrialStatus.RUNNING:
            self.ongoing_trials[tuner_id] = trial
            self.trials[trial_id] = trial

        return trial

    def update_trial(self, trial_id, metrics, step=0):
        """Used by a worker to report the status of a trial.

        Args:
            trial_id: A previously seen trial id.
            metrics: Dict of float. The current value of this
                trial's metrics.
            step: (Optional) Float. Used to report intermediate results. The
                current value in a timeseries representing the state of the
                trial. This is the value that `metrics` will be associated with.

        Returns:
            Trial object. Trial.status will be set to "STOPPED" if the Trial
            should be stopped early.
        """
        trial = self.trials[trial_id]
        self._check_objective_found(metrics)
        for metric_name, metric_value in metrics.items():
            trial.metrics.update(metric_name, metric_value, step=step)
        # To signal early stopping, set Trial.status to "STOPPED".
        return trial.status

    def end_trial(self, trial_id, status="COMPLETED"):
        """Record the measured objective for a set of parameter values.

        Args:
            trial_id: String. Unique id for this trial.
            status: String, one of "COMPLETED", "INVALID". A status of
                "INVALID" means a trial has crashed or been deemed
                infeasible.
        """
        trial = None
        for tuner_id, ongoing_trial in self.ongoing_trials.items():
            if ongoing_trial.trial_id == trial_id:
                trial = self.ongoing_trials.pop(tuner_id)
                break

        if not trial:
            raise ValueError(
                'Ongoing trial with id: {} not found.'.format(trial_id))

        trial.status = status
        if status == trial_lib.TrialStatus.COMPLETED:
            self._score_trial(trial)

    def _score_trial(self, trial):
        # Assumes single objective, subclasses can override.
        trial.score = trial.metrics.get_best_value(self.objective.name)
        trial.best_step = trial.metrics.get_best_step(self.objective.name)

    def get_trial(self, trial_id):
        return self.trials[trial_id]

    def get_best_trials(self, num_trials=1):
        trials = [t for t in self.trials.values()
                  if t.status == trial_lib.TrialStatus.COMPLETED]

        sorted_trials = sorted(
            trials,
            key=lambda trial: trial.score,
            # Assumes single objective, subclasses can override.
            reverse=self.objective.direction == 'max'
        )

        if self.executions_per_trial == 1:
            return sorted_trials[:num_trials]

        # Filter out Trials with identical hyperparameters.
        # Return the best Trial for each set of hyperparameters.
        return_trials = []
        seen_hps = []
        for trial in sorted_trials:
            if trial.hyperparameters.values not in seen_hps:
                return_trials.append(trial)
            if len(return_trials) == num_trials:
                break
        return return_trials

    def remaining_trials(self):
        if self.max_trials:
            return self.max_trials - len(self.trials.items())
        else:
            return None

    def get_state(self):
        state = {}
        state['trials'] = {trial_id: trial.get_state()
                           for trial_id, trial in self.trials.items()}
        # Just save the IDs for ongoing trials, since these are in `trials`.
        state['ongoing_trials'] = {
            tuner_id: trial.trial_id
            for tuner_id, trial in self.ongoing_trials.items()}
        state['last_trial_execution'] = self._last_trial_execution
        state['last_hps'] = self._last_hps
        return state

    def set_state(self, state):
        self.trials = {
            trial_id: trial_lib.Trial.from_state(trial_config)
            for trial_id, trial_config in state['trials'].items()}
        self.ongoing_trials = {
            tuner_id: self.trials[trial_id]
            for tuner_id, trial_id in state['ongoing_trials'].items()}
        self._last_trial_execution = state['last_trial_execution']
        self._last_hps = state['last_hps']

    def _compute_values_hash(self, values):
        keys = sorted(values.keys())
        s = ''.join(str(k) + '=' + str(values[k]) for k in keys)
        return hashlib.sha256(s.encode('utf-8')).hexdigest()[:32]

    def _check_objective_found(self, metrics):
        if isinstance(self.objective, Objective):
            objective_names = [self.objective.name]
        else:
            objective_names = [obj.name for obj in self.objective]
        for metric_name in metrics.keys():
            if metric_name in objective_names:
                objective_names.remove(metric_name)
        if objective_names:
            raise ValueError(
                'Objective value missing in metrics reported to the '
                'Oracle, expected: {}, found: {}'.format(
                    objective_names, metrics.keys()))


def _format_objective(objective):
    if isinstance(objective, (list, tuple)):
        return [_format_objective(obj for obj in objective)]

    if isinstance(objective, Objective):
        return objective
    if isinstance(objective, str):
        direction = metrics_tracking.infer_metric_direction(objective)
        return Objective(name=objective, direction=direction)
    else:
        raise ValueError('`objective` not understood, expected str or '
                         '`Objective` object, found: {}'.format(objective))
