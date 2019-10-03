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

from concurrent import futures
import grpc
import os
import time

from ..engine import hyperparameters as hp_module
from ..protos import service_pb2
from ..protos import service_pb2_grpc


class OracleServicer(service_pb2_grpc.OracleServicer):

    def __init__(self, oracle):
        self.oracle = oracle

    def GetSpace(self, request, context):
        hps = self.oracle.get_space()
        return service_pb2.GetSpaceResponse(
            hyperparameters=hps.to_proto())

    def UpdateSpace(self, request, context):
        hps = hp_module.HyperParameters.from_proto(
            request.hyperparameters)
        self.oracle.update_space(hps)
        return service_pb2.UpdateSpaceResponse()

    def CreateTrial(self, request, context):
        trial = self.oracle.create_trial(request.tuner_id)
        return service_pb2.CreateTrialResponse(
            trial=trial.to_proto())


def start_servicer(oracle):
    """Starts the `OracleServicer` used to manage distributed requests."""
    ip_addr = os.environ['KERASTUNER_ORACLE_IP']
    port = os.environ['KERASTUNER_ORACLE_PORT']
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=10))
    service_pb2_grpc.add_OracleServicer_to_server(
        OracleServicer(oracle), server)
    server.add_insecure_port('{}:{}'.format(ip_addr, port))
    server.start()
    while True:
        # The server does not block.
        time.sleep(10)
