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

from unittest import mock

import numpy as np
import pytest
import tensorflow as tf

import kerastuner as kt
from kerastuner.engine import hyperparameters
from kerastuner.engine import trial as trial_module
from kerastuner.engine import hyperparameters as hp_module
from kerastuner.tuners import hyperband as hyperband_module


@pytest.fixture(scope='function')
def tmp_dir(tmpdir_factory):
    return tmpdir_factory.mktemp('hyperband_test', numbered=True)


def test_hyperband_oracle_bracket_configs(tmp_dir):
    oracle = hyperband_module.HyperbandOracle(
        objective='score',
        max_sweeps=1,
        max_epochs=8,
        min_epochs=1,
        factor=2)
    oracle._set_project_dir(tmp_dir, 'untitled')

    # 8, 4, 2, 1 starting epochs.
    assert oracle._get_num_brackets() == 4

    assert oracle._get_num_rounds(bracket_num=3) == 4
    assert oracle._get_size(bracket_num=3, round_num=0) == 8
    assert oracle._get_epochs(bracket_num=3, round_num=0) == 1
    assert oracle._get_size(bracket_num=3, round_num=3) == 1
    assert oracle._get_epochs(bracket_num=3, round_num=3) == 8

    assert oracle._get_num_rounds(bracket_num=0) == 1
    assert oracle._get_size(bracket_num=0, round_num=0) == 4
    assert oracle._get_epochs(bracket_num=0, round_num=0) == 8


def test_hyperband_oracle_populate_brackets(tmp_dir):
    hp = kt.HyperParameters()
    hp.Float('a', -100, 100)
    hp.Float('b', -100, 100)
    oracle = hyperband_module.HyperbandOracle(
        hyperparameters=hp,
        objective='score',
        max_sweeps=1,
        max_epochs=9,
        min_epochs=1,
        factor=3)
    oracle._set_project_dir(tmp_dir, 'untitled')

    assert oracle._get_num_brackets() == 3

    assert oracle._get_size(2, 0) == 9
    b2_trials = []
    for i in range(9):
        trial = oracle.create_trial('b2' + str(i))
        assert trial.status == 'RUNNING'
        b2_trials.append(trial)

    assert len(oracle._brackets) == 1
    assert oracle._brackets[0]['bracket_num'] == 2
    assert len(oracle._brackets[0]['rounds'][0]) == 9

    # 2nd round from original bracket can't start, so run trials
    # for the next bracket.
    assert oracle._get_size(1, 0) == 5
    b1_trials = []
    for i in range(5):
        trial = oracle.create_trial('b1' + str(i))
        assert trial.status == 'RUNNING', i
        b1_trials.append(trial)

    assert len(oracle._brackets) == 2
    assert oracle._brackets[1]['bracket_num'] == 1
    assert len(oracle._brackets[1]['rounds'][0]) == 5

    # Finish enough trials from original bracket for its next
    # round to start.
    thrown_away_round_2 = 9 - 3
    for i in range(thrown_away_round_2 + 1):
        trial_id = b2_trials[i].trial_id
        oracle.update_trial(trial_id, {'score': i})
        oracle.end_trial(trial_id, 'COMPLETED')

    round2_trial = oracle.create_trial('r2')
    assert len(oracle._brackets) == 2
    assert len(oracle._brackets[0]['rounds'][0]) == 9
    assert len(oracle._brackets[0]['rounds'][1]) == 1


