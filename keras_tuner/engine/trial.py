# Copyright 2019 The KerasTuner Authors
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


import hashlib
import random
import time

import tensorflow as tf

from keras_tuner.engine import hyperparameters as hp_module
from keras_tuner.engine import metrics_tracking
from keras_tuner.engine import stateful
from keras_tuner.protos import keras_tuner_pb2


class TrialStatus:
    RUNNING = "RUNNING"
    INVALID = "INVALID"
    STOPPED = "STOPPED"
    COMPLETED = "COMPLETED"

    @staticmethod
    def to_proto(status):
        ts = keras_tuner_pb2.TrialStatus
        if status is None:
            return ts.TRIAL_UNKNOWN
        elif status == TrialStatus.RUNNING:
            return ts.TRIAL_RUNNING
        elif status == TrialStatus.INVALID:
            return ts.TRIAL_INVALID
        elif status == TrialStatus.STOPPED:
            return ts.TRIAL_STOPPED
        elif status == TrialStatus.COMPLETED:
            return ts.TRIAL_COMPLETED
        else:
            raise ValueError(f"Unknown status {status}")

    @staticmethod
    def from_proto(proto):
        ts = keras_tuner_pb2.TrialStatus
        if proto == ts.TRIAL_UNKNOWN:
            return None
        elif proto == ts.TRIAL_RUNNING:
            return TrialStatus.RUNNING
        elif proto == ts.TRIAL_INVALID:
            return TrialStatus.INVALID
        elif proto == ts.TRIAL_STOPPED:
            return TrialStatus.STOPPED
        elif proto == ts.TRIAL_COMPLETED:
            return TrialStatus.COMPLETED
        else:
            raise ValueError(f"Unknown status {proto}")


class Trial(stateful.Stateful):
    """The runs with the same set of hyperparameter values.

    Args:
        hyperparameters: HyperParameters. It contains the hyperparameter values
            for the trial.
        trial_id: String. The unique identifier for a trial.
        status: one of the OracleStatus attributes. It marks the current status
            of the Oracle.
    """

    def __init__(self, hyperparameters, trial_id=None, status=TrialStatus.RUNNING):
        self.hyperparameters = hyperparameters
        self.trial_id = generate_trial_id() if trial_id is None else trial_id

        self.metrics = metrics_tracking.MetricsTracker()
        self.score = None
        self.best_step = 0
        self.status = status

    def summary(self):
        """Displays a summary of this Trial."""
        print("Trial summary")

        print("Hyperparameters:")
        self.display_hyperparameters()

        if self.score is not None:
            print(f"Score: {self.score}")

    def display_hyperparameters(self):
        if self.hyperparameters.values:
            for hp, value in self.hyperparameters.values.items():
                print(hp + ":", value)
        else:
            print("default configuration")

    def get_state(self):
        return {
            "trial_id": self.trial_id,
            "hyperparameters": self.hyperparameters.get_config(),
            "metrics": self.metrics.get_config(),
            "score": self.score,
            "best_step": self.best_step,
            "status": self.status,
        }

    def set_state(self, state):
        self.trial_id = state["trial_id"]
        hp = hp_module.HyperParameters.from_config(state["hyperparameters"])
        self.hyperparameters = hp
        self.metrics = metrics_tracking.MetricsTracker.from_config(state["metrics"])
        self.score = state["score"]
        self.best_step = state["best_step"]
        self.status = state["status"]

    @classmethod
    def from_state(cls, state):
        trial = cls(hyperparameters=None)
        trial.set_state(state)
        return trial

    @classmethod
    def load(cls, fname):
        with tf.io.gfile.GFile(fname, "r") as f:
            state_data = f.read()
        return cls.from_state(state_data)

    def to_proto(self):
        if self.score is not None:
            score = keras_tuner_pb2.Trial.Score(
                value=self.score, step=self.best_step
            )
        else:
            score = None
        proto = keras_tuner_pb2.Trial(
            trial_id=self.trial_id,
            hyperparameters=self.hyperparameters.to_proto(),
            score=score,
            status=TrialStatus.to_proto(self.status),
            metrics=self.metrics.to_proto(),
        )
        return proto

    @classmethod
    def from_proto(cls, proto):
        instance = cls(
            hp_module.HyperParameters.from_proto(proto.hyperparameters),
            trial_id=proto.trial_id,
            status=TrialStatus.from_proto(proto.status),
        )
        if proto.HasField("score"):
            instance.score = proto.score.value
            instance.best_step = proto.score.step
        instance.metrics = metrics_tracking.MetricsTracker.from_proto(proto.metrics)
        return instance


def generate_trial_id():
    s = str(time.time()) + str(random.randint(1, int(1e7)))
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:32]
