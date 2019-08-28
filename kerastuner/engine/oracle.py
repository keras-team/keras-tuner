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


class Oracle(object):

    def __init__(self):
        self.space = []

    def update_space(self, new_entries):
        """Add new hyperparameters to the tracking space.

        Already recorded parameters get ignored.

        Args:
            new_entries: A list of HyperParameter objects to track.
        """
        ref_names = {p.name for p in self.space}
        for p in new_entries:
            if p.name not in ref_names:
                self.space.append(p)

    def populate_space(self, trial_id, space):
        """Fill a given hyperparameter space with values.

        Args:
            trial_id: String. Unique id for this trial.
            space: A list of HyperParameter objects
                to provide values for.

        Returns:
            A dictionary with keys:
                - status: string, one of "RUN", "IDLE", "EXIT"
                - values: Only included if status is "RUN".
                    Dict mapping parameter names to suggested values.
                    Note that if the Oracle is keeping tracking of a large
                    space, it may return values for more parameters
                    than what was listed in `space`.
        """
        raise NotImplementedError

    def result(self, trial_id, score):
        """Record the measured objective for a set of parameter values.

        If not overridden, this method does nothing.

        Args:
            trial_id: String. Unique id for this trial.
            score: Scalar. Lower is better.
        """
        pass

    def report_status(self, trial_id, status, score=None, t=None):
        """Used by a worker to report the current status of a trial.

        Args:
            trial_id: A previously seen trial id.
            status: String, one of "RUNNING", "CANCELLED". A status of
                "CANCELLED" will be passed when a trial has crashed or been
                deemed infeasible.
            score: (Optional) Float. The current, intermediate value of the
                trial's objective.
            t: (Optional) Float. The current value in a timeseries representing
                the state of the trial. This is the value that `score` will be
                associated with.

        Returns:
            `OracleResponse.STOP` if the trial should be stopped, otherwise 
            `OracleResponse.OK`.
        """
        return OracleResponse.OK

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
