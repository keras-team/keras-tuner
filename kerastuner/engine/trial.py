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
"""Trial class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import json

from . import metrics_tracking
from . import hyperparameters as hp_module
from . import execution as execution_module
from ..abstractions import display
from ..abstractions.tensorflow import TENSORFLOW_UTILS as tf_utils


class Trial(object):

    def __init__(self,
                 trial_id,
                 hyperparameters,
                 max_executions,
                 base_directory='.'):
        self.trial_id = trial_id
        self.hyperparameters = hyperparameters
        self.max_executions = max_executions
        self.base_directory = base_directory

        self.averaged_metrics = metrics_tracking.MetricsTracker()
        self.score = None
        self.executions = []

        self.directory = os.path.join(base_directory, 'trial_' + trial_id)
        tf_utils.create_directory(self.directory)

    def summary(self):
        display.section('Trial summary')
        if self.hyperparameters.values:
            display.subsection('Hp values:')
            display.display_settings(self.hyperparameters.values)
        else:
            display.subsection('Hp values: default configuration.')
        if self.score:
            display.display_setting('Score: %.4f' % self.score)

    def get_status(self):
        return {
            'trial_id': self.trial_id,
            'hyperparameters': self.hyperparameters.get_config(),
            'executions_seen': len(self.executions),
            'max_executions': self.max_executions,
            'score': self.score
        }

    def save(self):
        state = {
            'trial_id': self.trial_id,
            'hyperparameters': self.hyperparameters.get_config(),
            'max_executions': self.max_executions,
            'base_directory': self.base_directory,
            'averaged_metrics': self.averaged_metrics.get_config(),
            'score': self.score,
            'executions': [
                e.save() for e in self.executions
                if e.training_complete]
        }
        state_json = json.dumps(state)
        fname = os.path.join(self.directory, 'execution.json')
        tf_utils.write_file(fname, state_json)
        return fname

    @classmethod
    def load(cls, fname):
        state = json.load(fname)
        hp = hp_module.HyperParameters.from_config(
            state['hyperparameters']
        )
        trial = cls(
            trial_id=state['trial_id'],
            hyperparameters=hp,
            max_executions=state['max_executions'],
            base_directory=state['base_directory']
        )
        trial.score = state['score']
        metrics = metrics_tracking.MetricsTracker.from_config(
            [state['averaged_metrics']])
        trial.averaged_metrics = metrics
        trial.executions = [
            execution_module.Execution.load(f)
            for f in state['executions']
        ]
        return trial
