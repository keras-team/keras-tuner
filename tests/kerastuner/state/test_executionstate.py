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

from __future__ import absolute_import

import json
import pytest
from .common import is_serializable

from kerastuner.abstractions.tensorflow import TENSORFLOW_UTILS as tf_utils
from kerastuner.states.executionstate import ExecutionState


@pytest.fixture(scope='class')
def metric_config():
    return [{
        "statistics": {
            "min": 0.5,
            "max": 0.5,
            "median": 0.5,
            "mean": 0.5,
            "variance": 0.0,
            "stddev": 0.0
        },
        "history": [0.5, 0.5, 0.5],
        "direction": "max",
        "is_objective": False,
        "best_value": 0.5,
        "name": "accuracy",
        "start_time": 1234,
        "last_value": 0.5,
        "wall_time": [3, 2, 1]}]


def test_is_serializable(metric_config):
    state = ExecutionState(3, metric_config)
    is_serializable(state)


def test_from_config(metric_config):

    state = ExecutionState(3, metric_config)
    state2 = state.from_config(state.get_config())

    assert state.max_epochs == state2.max_epochs
    assert state.epochs == state2.epochs
    assert state.idx == state2.idx
    assert state.start_time == state2.start_time
    assert state.eta == state2.eta
    assert json.dumps(state.metrics.get_config()) == json.dumps(
        state2.metrics.get_config())
