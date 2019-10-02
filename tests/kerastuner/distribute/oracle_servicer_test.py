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

from concurrent import futures
import grpc
import os
import time

import kerastuner as kt
from kerastuner.distribute import oracle_servicer
from kerastuner.protos import service_pb2
from kerastuner.protos import service_pb2_grpc
from kerastuner.tuners import randomsearch
from .. import mock_distribute


def create_stub():
    # Give the OracleServicer time to come on-line.
    time.sleep(2)
    ip_addr = os.environ['KERASTUNER_ORACLE_IP']
    port = os.environ['KERASTUNER_ORACLE_PORT']
    channel = grpc.insecure_channel(
        '{}:{}'.format(ip_addr, port))
    return service_pb2_grpc.OracleStub(channel)


def test_get_space():

    def _test_get_space():
        tuner_id = os.environ['KERASTUNER_TUNER_ID']
        if 'chief' in tuner_id:
            hps = kt.HyperParameters()
            hps.Int('a', 0, 10, default=3)
            oracle = randomsearch.RandomSearchOracle(
                objective='score',
                max_trials=10,
                hyperparameters=hps)
            oracle_servicer.start_servicer(oracle)
        else:
            stub = create_stub()
            space_response = stub.GetSpace(service_pb2.GetSpaceRequest())
            retrieved_hps = kt.HyperParameters.from_proto(
                space_response.hyperparameters)
            assert retrieved_hps.values == {'a': 3}
            assert len(retrieved_hps.space) == 1

    mock_distribute.mock_distribute(_test_get_space)


def test_update_space():

    def _test_update_space():
        tuner_id = os.environ['KERASTUNER_TUNER_ID']
        if 'chief' in tuner_id:
            oracle = randomsearch.RandomSearchOracle(
                objective='score',
                max_trials=10)
            oracle_servicer.start_servicer(oracle)
        else:
            stub = create_stub()
            space_response = stub.GetSpace(service_pb2.GetSpaceRequest())
            retrieved_hps = kt.HyperParameters.from_proto(
                space_response.hyperparameters)
            assert len(retrieved_hps.space) == 0

            hps = kt.HyperParameters()
            hps.Int('a', 0, 10, default=5)
            hps.Choice('b', [1, 2, 3])
            request = service_pb2.UpdateSpaceRequest(
                hyperparameters=hps.to_proto())
            stub.UpdateSpace(request)

            space_response = stub.GetSpace(service_pb2.GetSpaceRequest())
            retrieved_hps = kt.HyperParameters.from_proto(
                space_response.hyperparameters)
            assert len(retrieved_hps.space) == 2
            assert retrieved_hps.values['a'] == 5
            assert retrieved_hps.values['b'] == 1

    mock_distribute.mock_distribute(_test_update_space)


def test_create_trial():

    def _test_create_trial():
        tuner_id = os.environ['KERASTUNER_TUNER_ID']
        if 'chief' in tuner_id:
            hps = kt.HyperParameters()
            hps.Int('a', 0, 10, default=5)
            hps.Choice('b', [1, 2, 3])
            oracle = randomsearch.RandomSearchOracle(
                objective='score',
                max_trials=10,
                hyperparameters=hps)
            oracle_servicer.start_servicer(oracle)
        else:
            stub = create_stub()
            response = stub.CreateTrial(service_pb2.CreateTrialRequest(
                tuner_id='worker0'))
            trial = kt.engine.trial.Trial.from_proto(response.trial)
            assert trial.status == "RUNNING"
            a = trial.hyperparameters.get('a')
            assert a >= 0 and a <= 10
            b = trial.hyperparameters.get('b')
            assert b in {1, 2, 3}

    mock_distribute.mock_distribute(_test_create_trial)
