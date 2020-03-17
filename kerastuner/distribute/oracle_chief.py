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
from ..engine import trial as trial_module
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
        return service_pb2.CreateTrialResponse(trial=trial.to_proto())

    def UpdateTrial(self, request, context):
        status = self.oracle.update_trial(request.trial_id,
                                          request.metrics,
                                          step=request.step)
        status_proto = trial_module._convert_trial_status_to_proto(status)
        return service_pb2.UpdateTrialResponse(status=status_proto)

    def EndTrial(self, request, context):
        status = trial_module._convert_trial_status_to_str(request.status)
        self.oracle.end_trial(request.trial_id, status)
        return service_pb2.EndTrialResponse()

    def GetTrial(self, request, context):
        trial = self.oracle.get_trial(request.trial_id)
        return service_pb2.GetTrialResponse(trial=trial.to_proto())

    def GetBestTrials(self, request, context):
        trials = self.oracle.get_best_trials(request.num_trials)
        return service_pb2.GetBestTrialsResponse(
            trials=[trial.to_proto() for trial in trials])


def start_server(oracle):
    """Starts the `OracleServicer` used to manage distributed requests."""
    ip_addr = os.environ['KERASTUNER_ORACLE_IP']
    port = os.environ['KERASTUNER_ORACLE_PORT']
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=1))
    service_pb2_grpc.add_OracleServicer_to_server(
        OracleServicer(oracle), server)
    server.add_insecure_port('{}:{}'.format(ip_addr, port))
    server.start()
    # since server.start() will not block,
    # a sleep-loop is added to keep alive
    try:
        while True:
            time.sleep(10)
    except KeyboardInterrupt:
        server.stop(0)
