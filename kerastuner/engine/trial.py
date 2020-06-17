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
import random
import tensorflow as tf
import time

from . import hyperparameters as hp_module
from . import metrics_tracking
from . import stateful
from ..protos import kerastuner_pb2


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
        """Displays a summary of this Trial."""
        print('Trial summary')

        print('Hyperparameters:')
        self.display_hyperparameters()

        if self.score is not None:
            print('Score: {}'.format(self.score))

    def display_hyperparameters(self):
        if self.hyperparameters.values:
            for hp, value in self.hyperparameters.values.items():
                print(hp + ':', value)
        else:
            print('default configuration')

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
        self.metrics = metrics_tracking.MetricsTracker.from_config(state['metrics'])
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
        with tf.io.gfile.GFile(fname, 'r') as f:
            state_data = f.read()
        return cls.from_state(state_data)

    def to_proto(self):
        if self.score is not None:
            score = kerastuner_pb2.Trial.Score(
                value=self.score, step=self.best_step)
        else:
            score = None
        proto = kerastuner_pb2.Trial(
            trial_id=self.trial_id,
            hyperparameters=self.hyperparameters.to_proto(),
            score=score,
            status=_convert_trial_status_to_proto(self.status),
            metrics=self.metrics.to_proto())
        return proto

    @classmethod
    def from_proto(cls, proto):
        instance = cls(
            hp_module.HyperParameters.from_proto(proto.hyperparameters),
            trial_id=proto.trial_id,
            status=_convert_trial_status_to_str(proto.status))
        if proto.HasField('score'):
            instance.score = proto.score.value
            instance.best_step = proto.score.step
        instance.metrics = metrics_tracking.MetricsTracker.from_proto(
            proto.metrics)
        return instance


def generate_trial_id():
    s = str(time.time()) + str(random.randint(1, 1e7))
    return hashlib.sha256(s.encode('utf-8')).hexdigest()[:32]


def _convert_trial_status_to_proto(status):
    ts = kerastuner_pb2.TrialStatus
    if status is None:
        return ts.UNKNOWN
    elif status == TrialStatus.RUNNING:
        return ts.RUNNING
    elif status == TrialStatus.IDLE:
        return ts.IDLE
    elif status == TrialStatus.INVALID:
        return ts.INVALID
    elif status == TrialStatus.STOPPED:
        return ts.STOPPED
    elif status == TrialStatus.COMPLETED:
        return ts.COMPLETED
    else:
        raise ValueError('Unknown status {}'.format(status))


def _convert_trial_status_to_str(status):
    ts = kerastuner_pb2.TrialStatus
    if status == ts.UNKNOWN:
        return None
    elif status == ts.RUNNING:
        return TrialStatus.RUNNING
    elif status == ts.IDLE:
        return TrialStatus.IDLE
    elif status == ts.INVALID:
        return TrialStatus.INVALID
    elif status == ts.STOPPED:
        return TrialStatus.STOPPED
    elif status == ts.COMPLETED:
        return TrialStatus.COMPLETED
    else:
        raise ValueError('Unknown status {}'.format(status))
