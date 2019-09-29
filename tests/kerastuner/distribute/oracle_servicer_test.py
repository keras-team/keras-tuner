from concurrent import futures
import grpc
import os
import time

import kerastuner as kt
from kerastuner.distribute import servicer
from kerastuner.protos import service_pb2
from kerastuner.protos import service_pb2_grpc
from kerastuner.tuners import randomsearch
from .. import mock_distribute


def start_servicer(oracle):
    port = os.environ['KERASTUNER_PORT']
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=10))
    service_pb2_grpc.add_OracleServicer_to_server(
        servicer.OracleServicer(oracle), server)
    server.add_insecure_port('127.0.0.1:{}'.format(port))
    server.start()
    while True:
        # The server does not block.
        time.sleep(10)


def create_stub():
    # Give the Servicer time to come on-line.
    time.sleep(2)
    port = os.environ['KERASTUNER_PORT']
    channel = grpc.insecure_channel(
        '127.0.0.1:{}'.format(port))
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
            start_servicer(oracle)
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
            start_servicer(oracle)
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
