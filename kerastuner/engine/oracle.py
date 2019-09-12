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

from tensorflow import keras
from kerastuner.engine import hyperparameters as hp_module
from kerastuner.engine import metrics_tracking
from kerastuner.engine import trial as trial_lib


Objective = collections.namedtuple('Objective', 'name direction')


class Oracle(object):

    def __init__(self,
                 objective,
                 max_trials,
                 hyperparameters=None,
                 allow_new_entries=True,
                 tune_new_entries=True):
        self.objective = _format_objective(objective)
        self.max_trials = max_trials
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

    def populate_space(self, trial_id):
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

        if len(self.trials.items()) >= self.max_trials:
            status = trial_lib.TrialStatus.STOPPED
            values = None
        else:
            response = self.populate_space(trial_id)
            status = response['status']
            values = response['values'] if 'values' in response else None

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

    def end_trial(self, trial_id, status):
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
            trial.score = self.score_trial(trial)

    def update_trial(self, trial_id, metrics, t=0):
        """Used by a worker to report the status of a trial.

        Args:
            trial_id: A previously seen trial id.
            metrics: Dict of float. The current value of this
                trial's metrics.
            t: (Optional) Float. Used to report intermediate results. The
                current value in a timeseries representing the state of the
                trial. This is the value that `metrics` will be associated with.

        Returns:
            Trial object. Trial.status will be set to "STOPPED" if the Trial
            should be stopped early.
        """
        trial = self.trials[trial_id]
        self._check_objective_found(metrics)
        for metric_name, metric_value in metrics.items():
            trial.metrics.update(metric_name, metric_value, t=t)
        # To handle early stopping, set Trial.status to "STOPPED".
        return trial

    def score_trial(self, trial):
        # Assumes single objective, subclasses can override.
        return metrics_tracking.MetricObservation(
            value=trial.metrics.get_best_value(self.objective.name),
            t=trial.metrics.get_best_t(self.objective.name))

    def get_trial(self, trial_id):
        return self.trials[trial_id]

    def get_best_trials(self, num_trials=1):
        trials = [t for t in self.trials.values()
                  if t.status == trial_lib.TrialStatus.COMPLETED]
        for trial in trials:
            trial.score = self.score_trial(trial)

        sorted_trials = sorted(
            trials,
            key=lambda trial: trial.score.value,
            # Assumes single objective, subclasses can override.
            reverse=self.objective.direction == 'max'
        )
        return sorted_trials[:num_trials]

    def remaining_trials(self):
        return self.max_trials - len(self.trials.items())

    def save(self, fname):
        # TODO: Save completed trials.
        raise NotImplementedError

    def reload(self, fname):
        # TODO: Restore completed trials.
        raise NotImplementedError

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
                'Objective value missing in metrics reported '
                'to the Oracle, expected: {}'.format(
                    objective_names))


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
