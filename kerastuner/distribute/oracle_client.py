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
"""OracleClient class."""

import grpc
import os
import time

from ..engine import hyperparameters as hp_module
from ..engine import trial as trial_module
from ..protos import service_pb2
from ..protos import service_pb2_grpc


class OracleClient(object):
    """Wraps an `Oracle` on a worker to send requests to the chief."""

    def __init__(self, oracle):
        self._oracle = oracle

        # Allow time for the OracleServicer to come on-line.
        time.sleep(3)
        ip_addr = os.environ['KERASTUNER_ORACLE_IP']
        port = os.environ['KERASTUNER_ORACLE_PORT']
        channel = grpc.insecure_channel(
            '{}:{}'.format(ip_addr, port))
        self.stub = service_pb2_grpc.OracleStub(channel)

    def __getattr__(self, name):
        whitelisted_attrs = {
            'objective',
            'max_trials',
            'allow_new_entries',
            'tune_new_entries'}
        if name in whitelisted_attrs:
            return getattr(self._oracle, name)
        return super(OracleClient, self).__getattr__(name)

    def get_space(self):
        response = self.stub.GetSpace(service_pb2.GetSpaceRequest())
        return hp_module.HyperParameters.from_proto(response.hyperparameters)

    def update_space(self, hyperparameters):
        self.stub.UpdateSpace(service_pb2.UpdateSpaceRequest(
            hyperparameters=hyperparameters.to_proto()))

    def create_trial(self, tuner_id):
        response = self.stub.CreateTrial(service_pb2.CreateTrialRequest(
            tuner_id=tuner_id))
        return trial_module.Trial.from_proto(response.trial)

    def update_trial(self, trial_id, metrics, step=0):
        response = self.stud.UpdateTrial(service_pb2.UpdateTrialRequest(
            trial_id=trial_id, metrics=metrics, step=step))
        return trial_module._convert_trial_status_to_str(response.trial_status)
