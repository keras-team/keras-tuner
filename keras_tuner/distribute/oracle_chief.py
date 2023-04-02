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

import os
import time
from concurrent import futures

import grpc

from keras_tuner import protos
from keras_tuner.engine import hyperparameters as hp_module
from keras_tuner.engine import trial as trial_module


class OracleServicer(protos.get_service_grpc().OracleServicer):
    def __init__(self, oracle):
        self.oracle = oracle
        self.stop_triggered = False

    def GetSpace(self, request, context):
        hps = self.oracle.get_space()
        return protos.get_service().GetSpaceResponse(
            hyperparameters=hps.to_proto()
        )

    def UpdateSpace(self, request, context):
        hps = hp_module.HyperParameters.from_proto(request.hyperparameters)
        self.oracle.update_space(hps)
        return protos.get_service().UpdateSpaceResponse()

    def CreateTrial(self, request, context):
        trial = self.oracle.create_trial(request.tuner_id)
        if trial.status == trial_module.TrialStatus.STOPPED:
            self.stop_triggered = True
        return protos.get_service().CreateTrialResponse(trial=trial.to_proto())

    def UpdateTrial(self, request, context):
        trial = self.oracle.update_trial(
            request.trial_id, request.metrics, step=request.step
        )
        return protos.get_service().UpdateTrialResponse(trial=trial.to_proto())

    def EndTrial(self, request, context):
        trial = trial_module.Trial.from_proto(request.trial)
        self.oracle.end_trial(trial)
        return protos.get_service().EndTrialResponse()

    def GetTrial(self, request, context):
        trial = self.oracle.get_trial(request.trial_id)
        return protos.get_service().GetTrialResponse(trial=trial.to_proto())

    def GetBestTrials(self, request, context):
        trials = self.oracle.get_best_trials(request.num_trials)
        return protos.get_service().GetBestTrialsResponse(
            trials=[trial.to_proto() for trial in trials]
        )


def start_server(oracle):
    """Starts the `OracleServicer` used to manage distributed requests."""
    ip_addr = os.environ["KERASTUNER_ORACLE_IP"]
    port = os.environ["KERASTUNER_ORACLE_PORT"]
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
    oracle_servicer = OracleServicer(oracle)
    protos.get_service_grpc().add_OracleServicer_to_server(
        oracle_servicer, server
    )
    server.add_insecure_port(f"{ip_addr}:{port}")
    server.start()
    while True:
        # The server does not block otherwise.
        time.sleep(30)

        if oracle_servicer.stop_triggered:
            while oracle.ongoing_trials:
                time.sleep(10)  # pragma: no cover

            print(
                "Oracle server on chief is exiting in 40s."
                "The chief will go on with post-search code."
            )
            # Wait for another 30s for all the subsequent calls from workers to
            # come
            time.sleep(30)
            server.stop(10)
            break
