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

import pytest
import os
from pathlib import Path

from kerastuner.collections.instancestatescollection import InstanceStatesCollection


@pytest.fixture()
def data_path():
    data_path = Path(__file__).parents[2]
    data_path = Path.joinpath(data_path, 'data', 'tuner_results')
    return data_path


@pytest.fixture()
def col(data_path):
    col = InstanceStatesCollection()
    count = col.load_from_dir(data_path)
    assert count == 6
    return col


def test_loading(data_path):
    col = InstanceStatesCollection()
    count = col.load_from_dir(data_path)
    assert count == 6


def test_loading_objective_canonicalization(col):
    instance_state = col.get_last()
    assert instance_state.objective == 'val_accuracy'


def test_sort_by_metrics(col):
    instance_states = col.sort_by_metric('loss')
    losses = []
    accuracies = []
    for instance_state in instance_states:
        l = instance_state.agg_metrics.get('loss').get_best_value()
        losses.append(l)
        a = instance_state.agg_metrics.get(
            'val_accuracy').get_best_value()
        accuracies.append(a)
    assert losses[0] < losses[1]
    assert accuracies[0] < accuracies[1]
    assert len(accuracies) == 6
