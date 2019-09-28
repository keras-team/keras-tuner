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

import pytest
import numpy as np

import kerastuner
from kerastuner.engine import base_tuner

import tensorflow as tf
from tensorflow import keras


@pytest.fixture(scope='module')
def tmp_dir(tmpdir_factory):
    return tmpdir_factory.mktemp('integration_test')


def test_base_tuner(tmp_dir):
    class MyTuner(base_tuner.BaseTuner):
        def run_trial(self, trial, x):
            model = self.hypermodel.build(trial.hyperparameters)
            self.oracle.update_space(trial.hyperparameters)
            score = model(x)
            self.oracle.update_trial(
                trial.trial_id, metrics={'score': score})

        def get_best_models(self, num_models=1):
            best_trials = self.oracle.get_best_trials(num_models)
            models = [self.hypermodel.build(t.hyperparameters)
                      for t in best_trials]
            return models

    def build_model(hp):
        class MyModel(object):
            def __init__(self):
                self.factor = hp.Float('a', 0, 10)

            def __call__(self, x):
                return self.factor * x
        return MyModel()

    oracle = kerastuner.tuners.randomsearch.RandomSearchOracle(
        objective=kerastuner.Objective('score', 'max'),
        max_trials=5)
    tuner = MyTuner(
        oracle=oracle,
        hypermodel=build_model,
        directory=tmp_dir)
    tuner.search(1.0)
    models = tuner.get_best_models(5)

    # Check that scoring of the model was done correctly.
    models_by_factor = sorted(models,
                              key=lambda m: m.factor,
                              reverse=True)
    assert models[0] == models_by_factor[0]

