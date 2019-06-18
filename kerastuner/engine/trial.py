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

from . import metrics_tracking
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

        self.averaged_metrics = metrics_tracking.MetricsTracker()
        self.score = None
        self.executions = []
        self.directory = os.path.join(base_directory, 'trial_' + trial_id)
        tf_utils.create_directory(self.directory)

    def summary(self):
        display.section('Trial summary')
        display.subsection('Hp values:')
        display.display_settings(self.hyperparameters.values)
        if self.score:
            display.display_setting('Score: %.4f' % self.score)

    def get_status(self):
        return {
            'trial_id': self.trial_id,
            'hyperparameters': self.hyperparameters.get_config(),
            'executions_seen': len(self.executions),
            'max_executions': self.max_executions,
            'averaged_metrics': self.averaged_metrics.get_config(),
            'score': self.score
        }
