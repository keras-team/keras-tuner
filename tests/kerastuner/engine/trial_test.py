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

from kerastuner.engine import hyperparameters as hp_module
from kerastuner.engine import metrics_tracking
from kerastuner.engine import trial as trial_module


def test_trial_proto():
    hps = hp_module.HyperParameters()
    hps.Int('a', 0, 10, default=3)
    trial = trial_module.Trial(
        hps, trial_id='trial1', status='COMPLETED')
    trial.metrics.register('score', direction='max')
    trial.metrics.update('score', 10, step=1)

    proto = trial.to_proto()
    assert len(proto.hyperparameters.space.int_space) == 1
    assert proto.hyperparameters.values.values['a'].int_value == 3
    assert not proto.HasField('score')

    new_trial = trial_module.Trial.from_proto(proto)
    assert new_trial.status == 'COMPLETED'
    assert new_trial.hyperparameters.get('a') == 3
    assert new_trial.trial_id == 'trial1'
    assert new_trial.score is None
    assert new_trial.best_step is None

    trial.score = -10
    trial.best_step = 3

    proto = trial.to_proto()
    assert proto.HasField('score')
    assert proto.score.value == -10
    assert proto.score.step == 3

    new_trial = trial_module.Trial.from_proto(proto)
    assert new_trial.score == -10
    assert new_trial.best_step == 3
    assert new_trial.metrics.get_history('score') == [
        metrics_tracking.MetricObservation(10, step=1)]
