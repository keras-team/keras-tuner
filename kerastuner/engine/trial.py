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

import hashlib
import json
import os
import random
import time

from kerastuner.engine import stateful
from . import metrics_tracking
from . import hyperparameters as hp_module
from ..abstractions import display
from ..abstractions.tensorflow import TENSORFLOW_UTILS as tf_utils


class TrialStatus:
    RUNNING = 'RUNNING'
    IDLE = 'IDLE'
    INVALID = 'INVALID'
    STOPPED = 'STOPPED'
    COMPLETED = 'COMPLETED'


class Trial(stateful.Stateful):

    def __init__(self,
                 hyperparameters,
                 trial_id=None,
                 status=TrialStatus.RUNNING):
        self.hyperparameters = hyperparameters
        self.trial_id = generate_trial_id() if trial_id is None else trial_id

        self.metrics = metrics_tracking.MetricsTracker()
        self.score = None
        self.best_step = None
        self.status = status

    def summary(self):
        display.section('Trial summary')
        if self.hyperparameters.values:
            display.subsection('Hp values:')
            display.display_settings(self.hyperparameters.values)
        else:
            display.subsection('Hp values: default configuration.')
        if self.score is not None:
            display.display_setting('Score: {}'.format(self.score))
        if self.best_step is not None:
            display.display_setting('Best step: {}'.format(self.best_step))

    def get_state(self):
        return {
            'trial_id': self.trial_id,
            'hyperparameters': self.hyperparameters.get_config(),
            'metrics': self.metrics.get_config(),
            'score': self.score,
            'best_step': self.best_step,
            'status': self.status
        }

    def set_state(self, state):
        self.trial_id = state['trial_id']
        hp = hp_module.HyperParameters.from_config(
            state['hyperparameters']
        )
        self.hyperparameters = hp
        metrics = metrics_tracking.MetricsTracker.from_config(
            state['metrics'])
        self.score = state['score']
        self.best_step = state['best_step']
        self.status = state['status']

    @classmethod
    def from_state(cls, state):
        trial = cls(hyperparameters=None)
        trial.set_state(state)
        return trial

    @classmethod
    def load(cls, fname):
        state_data = tf_utils.read_file(fname)
        state = json.loads(state_data)
        return cls.from_state(state_data)


def generate_trial_id():
    s = str(time.time()) + str(random.randint(1, 1e7))
    return hashlib.sha256(s.encode('utf-8')).hexdigest()[:32]
