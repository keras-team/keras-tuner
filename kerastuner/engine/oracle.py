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

import enum
import hashlib

from kerastuner.engine import trial as trial_lib


class Oracle(object):

    def __init__(self,
                 objective,
                 max_trials,
                 hyperparameters=None,
                 allow_new_entries=True,
                 tune_new_entries=True):
        self.objective = objective
        self.max_trials = max_trials
        self.hyperparameters = hyperparameters
        self.allow_new_entries = allow_new_entries
        self.tune_new_entries = tune_new_entries
        # trial_id -> Trial
        self.trials = {}
        # tuner_id -> Trial
        self.ongoing_trials = {}

    def get_space(self):
        return self.hyperparameters

    def update_space(self, new_entries):
        """Add new hyperparameters to the tracking space.

        Already recorded parameters get ignored.

        Args:
            new_entries: A list of HyperParameter objects to track.
        """
        ref_names = {hp.name for hp in self.hyperparameters.space}
        new_hps = [hp for hp in new_entries if hp.name not in ref_names]

        if new_hps and not self.allow_new_entries:
            raise ValueError('`allow_new_entries` is `False`, but found '
                             'new entries {}'.format(new_hps))

        if not self.tune_new_entries:
            # New entries should always use the default value.
            return

        for hp in new_hps:
            self.hyperparameters.register(
                hp.name, type(hp), hp.get_config())

    def populate_space(self):
        """Fill the hyperparameter space with values for a trial.

        Returns:
          HyperParameters object.
        """
        raise NotImplementedError

    def create_trial(self, tuner_id):
        # Allow for multi-worker DistributionStrategy within a Trial.
        if tuner_id in self.ongoing_trials:
            return self.ongoing_trials[tuner_id]

        trial = trial_lib.Trial(hyperparameters=self.populate_space())
        self.ongoing_trials[tuner_id] = trial
        self.trials[trial.trial_id] = trial
        return trial

    def end_trial(self, trial_id, status):
        """Record the measured objective for a set of parameter values.

        Args:
            trial_id: String. Unique id for this trial.
            status: String, one of "OK", "INVALID". A status of
                "INVALID" means a trial has crashed or been deemed
                infeasible.
        """
        trial = self.trials[trial_id]
        trial.status = status
        self.ongoing_trials.pop(trial)

    def update_trial(self, trial_id, metrics, t=None):
        """Used by a worker to report the status of a trial.

        Args:
            trial_id: A previously seen trial id.
            metrics: Dict of float. The current value of the
                trial's metrics.
            t: (Optional) Float. Used to report intermediate results. The
                current value in a timeseries representing the state of the
                trial. This is the value that `metrics` will be associated with.

        Returns:
            `OracleResponse.STOP` if the trial should be stopped, otherwise
            `OracleResponse.OK`.
        """
        trial = self.trials[trial_id]
        for metric_name, metric_value in metrics.items():
            # TODO: handle `t` in metrics tracker.
            trial.metrics.update(metric_name, metric_value)

    def get_best_trials(self, n):
        raise NotImplementedError

    def save(self, fname):
        raise NotImplementedError

    def reload(self, fname):
        raise NotImplementedError

    def _compute_values_hash(self, values):
        keys = sorted(values.keys())
        s = ''.join(str(k) + '=' + str(values[k]) for k in keys)
        return hashlib.sha256(s.encode('utf-8')).hexdigest()[:32]


class OracleResponse(enum.Enum):
    OK = 0
    STOP = 1
