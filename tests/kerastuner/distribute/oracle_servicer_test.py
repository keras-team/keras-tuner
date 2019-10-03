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
"""Tests for the OracleServicer class."""

import os

import kerastuner as kt
from kerastuner.distribute import oracle_client
from kerastuner.distribute import oracle_servicer
from kerastuner.tuners import randomsearch
from .. import mock_distribute


def test_get_space():

    def _test_get_space():
        hps = kt.HyperParameters()
        hps.Int('a', 0, 10, default=3)
        oracle = randomsearch.RandomSearchOracle(
            objective='score',
            max_trials=10,
            hyperparameters=hps)
        tuner_id = os.environ['KERASTUNER_TUNER_ID']
        if 'chief' in tuner_id:
            oracle_servicer.start_servicer(oracle)
        else:
            client = oracle_client.OracleClient(oracle)
            retrieved_hps = client.get_space()
            assert retrieved_hps.values == {'a': 3}
            assert len(retrieved_hps.space) == 1

    mock_distribute.mock_distribute(_test_get_space)


def test_update_space():

    def _test_update_space():
        oracle = randomsearch.RandomSearchOracle(
            objective='score',
            max_trials=10)
        tuner_id = os.environ['KERASTUNER_TUNER_ID']
        if 'chief' in tuner_id:
            oracle_servicer.start_servicer(oracle)
        else:
            client = oracle_client.OracleClient(oracle)

            hps = kt.HyperParameters()
            hps.Int('a', 0, 10, default=5)
            hps.Choice('b', [1, 2, 3])
            client.update_space(hps)

            retrieved_hps = client.get_space()
            assert len(retrieved_hps.space) == 2
            assert retrieved_hps.values['a'] == 5
            assert retrieved_hps.values['b'] == 1

    mock_distribute.mock_distribute(_test_update_space)


def test_create_trial():

    def _test_create_trial():
        hps = kt.HyperParameters()
        hps.Int('a', 0, 10, default=5)
        hps.Choice('b', [1, 2, 3])
        oracle = randomsearch.RandomSearchOracle(
            objective='score',
            max_trials=10,
            hyperparameters=hps)
        tuner_id = os.environ['KERASTUNER_TUNER_ID']
        if 'chief' in tuner_id:
            oracle_servicer.start_servicer(oracle)
        else:
            client = oracle_client.OracleClient(oracle)
            trial = client.create_trial(tuner_id)
            assert trial.status == "RUNNING"
            a = trial.hyperparameters.get('a')
            assert a >= 0 and a <= 10
            b = trial.hyperparameters.get('b')
            assert b in {1, 2, 3}

    mock_distribute.mock_distribute(_test_create_trial)
