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
"""Execution class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import os
import json

from . import metrics_tracking
from ..abstractions.tensorflow import TENSORFLOW_UTILS as tf_utils


class Execution(object):

    def __init__(self,
                 execution_id,
                 trial_id,
                 max_epochs,
                 max_steps,
                 base_directory='.'):
        self.execution_id = execution_id
        self.trial_id = trial_id
        self.max_epochs = max_epochs
        self.max_steps = max_steps
        self.base_directory = base_directory

        self.per_epoch_metrics = metrics_tracking.MetricsTracker()
        self.per_batch_metrics = metrics_tracking.MetricsTracker()
        self.epochs_seen = 0
        self.best_checkpoint = None
        self.training_complete = False
        self.start_time = int(time.time())

        self.directory = os.path.join(base_directory,
                                      'execution_' + execution_id)
        tf_utils.create_directory(self.directory)

    def get_state(self):
        return {
            'execution_id': self.execution_id,
            'trial_id': self.trial_id,
            'max_epochs': self.max_epochs,
            'max_steps': self.max_steps,
            'per_epoch_metrics': self.per_epoch_metrics.get_config(),
            'per_batch_metrics': self.per_batch_metrics.get_config(),
            'epochs_seen': self.epochs_seen,
            'best_checkpoint': self.best_checkpoint,
            'training_complete': self.training_complete,
            'start_time': self.start_time,
            'base_directory': self.base_directory,
            'eta': self.eta,
        }

    def save(self):
        state = self.get_state()
        state_json = json.dumps(state)
        fname = os.path.join(self.directory, 'execution.json')
        tf_utils.write_file(fname, state_json)
        return str(fname)

    @classmethod
    def load(cls, fname):
        state = json.load(fname)
        execution = cls(
            execution_id=state['execution_id'],
            trial_id=state['trial_id'],
            max_epochs=state['max_epochs'],
            max_steps=state['max_steps'],
            base_directory=state['base_directory']
        )
        per_epoch = metrics_tracking.MetricsTracker.from_config(
            state['per_epoch_metrics'])
        execution.per_epoch_metrics = per_epoch
        per_batch = metrics_tracking.MetricsTracker.from_config(
            state['per_batch_metrics'])
        execution.per_batch_metrics = per_batch
        execution.epochs_seen = state['epochs_seen']
        execution.best_checkpoint = state['best_checkpoint']
        execution.training_complete = state['training_complete']
        execution.start_time = state['start_time']
        return execution

    @property
    def eta(self):
        elapsed_time = int(time.time()) - self.start_time
        time_per_epoch = elapsed_time / max(self.epochs_seen, 1)
        return int(self.max_epochs * time_per_epoch)
