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

from ..engine import hyperparameters as hp_module
from ..engine import trial as trial_module
from ..protos import service_pb2
from ..protos import service_pb2_grpc


class OracleClient(object):
    """Wraps an `Oracle` on a worker to send requests to the chief."""

    def __init__(self, oracle):
        self._oracle = oracle

        ip_addr = os.environ['KERASTUNER_ORACLE_IP']
        port = os.environ['KERASTUNER_ORACLE_PORT']
        channel = grpc.insecure_channel(
            '{}:{}'.format(ip_addr, port))
        self.stub = service_pb2_grpc.OracleStub(channel)
        self.tuner_id = os.environ['KERASTUNER_TUNER_ID']

        # In multi-worker mode, only the chief of each cluster should report
        # results to the chief Oracle.
        self.multi_worker = False
        self.should_report = True

    def __getattr__(self, name):
        whitelisted_attrs = {
            'objective',
            'max_trials',
            'allow_new_entries',
            'tune_new_entries'}
        if name in whitelisted_attrs:
            return getattr(self._oracle, name)
        raise AttributeError(
            '`OracleClient` object has no attribute "{}"'.format(name))

    def get_space(self):
        response = self.stub.GetSpace(
            service_pb2.GetSpaceRequest(), wait_for_ready=True)
        return hp_module.HyperParameters.from_proto(response.hyperparameters)

    def update_space(self, hyperparameters):
        if self.should_report:
            self.stub.UpdateSpace(service_pb2.UpdateSpaceRequest(
                hyperparameters=hyperparameters.to_proto()),
                wait_for_ready=True)

    def create_trial(self, tuner_id):
        response = self.stub.CreateTrial(service_pb2.CreateTrialRequest(
            tuner_id=tuner_id), wait_for_ready=True)
        return trial_module.Trial.from_proto(response.trial)

    def update_trial(self, trial_id, metrics, step=0):
        # TODO: support early stopping in multi-worker.
        if self.should_report:
            response = self.stub.UpdateTrial(service_pb2.UpdateTrialRequest(
                trial_id=trial_id, metrics=metrics, step=step),
                wait_for_ready=True)
            if not self.multi_worker:
                return trial_module._convert_trial_status_to_str(response.status)
            return 'RUNNING'
        return 'RUNNING'

    def end_trial(self, trial_id, status="COMPLETED"):
        if self.should_report:
            status = trial_module._convert_trial_status_to_proto(status)
            self.stub.EndTrial(service_pb2.EndTrialRequest(
                trial_id=trial_id, status=status), wait_for_ready=True)

    def get_trial(self, trial_id):
        response = self.stub.GetTrial(service_pb2.GetTrialRequest(
            trial_id=trial_id), wait_for_ready=True)
        return trial_module.Trial.from_proto(response.trial)

    def get_best_trials(self, num_trials=1):
        response = self.stub.GetBestTrials(service_pb2.GetBestTrialsRequest(
            num_trials=num_trials), wait_for_ready=True)
        return [trial_module.Trial.from_proto(trial) for trial in response.trials]
