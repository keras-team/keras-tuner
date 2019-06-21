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
        self.current_epoch = 0

        self.per_epoch_metrics = metrics_tracking.MetricsTracker()
        self.per_batch_metrics = metrics_tracking.MetricsTracker()
        self.epochs_seen = 0
        self.best_checkpoint = None
        self.training_complete = False
        self.start_time = int(time.time())

        self.directory = os.path.join(base_directory,
                                      'execution_' + execution_id)
        tf_utils.create_directory(self.directory)

    def get_status(self):
        return {
            'execution_id': self.execution_id,
            'trial_id': self.trial_id,
            'start_time': self.start_time,
            'per_epoch_metrics': self.per_epoch_metrics.get_config(),
            'eta': self.eta,
            'epochs_seen': self.epochs_seen,
            'max_epochs': self.max_epochs,
            'training_complete': self.training_complete
        }

    @property
    def eta(self):
        elapsed_time = int(time.time()) - self.start_time
        time_per_epoch = elapsed_time / max(self.epochs_seen, 1)
        return int(self.max_epochs * time_per_epoch)
