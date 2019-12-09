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

import numpy as np
import os
import pickle
import pytest

import kerastuner
from kerastuner.engine import base_tuner

from sklearn import linear_model


INPUT_DIM = 2
NUM_CLASSES = 3
NUM_SAMPLES = 64
TRAIN_INPUTS = np.random.random(size=(NUM_SAMPLES, INPUT_DIM))
TRAIN_TARGETS = np.random.randint(0, NUM_CLASSES, size=(NUM_SAMPLES,))
VAL_INPUTS = np.random.random(size=(NUM_SAMPLES, INPUT_DIM))
VAL_TARGETS = np.random.randint(0, NUM_CLASSES, size=(NUM_SAMPLES,))


@pytest.fixture(scope='function')
def tmp_dir(tmpdir_factory):
    return tmpdir_factory.mktemp('integration_test', numbered=True)


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


def test_simple_sklearn_tuner(tmp_dir):
    class SimpleSklearnTuner(base_tuner.BaseTuner):
        def run_trial(self, trial, x, y, validation_data):
            model = self.hypermodel.build(trial.hyperparameters)
            model.fit(x, y)
            x_val, y_val = validation_data
            score = model.score(x_val, y_val)
            self.oracle.update_trial(
                trial.trial_id, {'score': score})
            self.save_model(trial.trial_id, model)

        def save_model(self, trial_id, model, step=0):
            fname = os.path.join(self.get_trial_dir(trial_id), 'model.pickle')
            with open(fname, 'wb') as f:
                pickle.dump(model, f)

        def load_model(self, trial):
            fname = os.path.join(
                self.get_trial_dir(trial.trial_id), 'model.pickle')
            with open(fname, 'rb') as f:
                return pickle.load(f)

    def sklearn_build_fn(hp):
        c = hp.Float('c', 1e-4, 10)
        return linear_model.LogisticRegression(C=c)

    tuner = SimpleSklearnTuner(
        oracle=kerastuner.tuners.randomsearch.RandomSearchOracle(
            objective=kerastuner.Objective('score', 'max'),
            max_trials=2),
        hypermodel=sklearn_build_fn,
        directory=tmp_dir)
    tuner.search(TRAIN_INPUTS,
                 TRAIN_TARGETS,
                 validation_data=(VAL_INPUTS, VAL_TARGETS))
    models = tuner.get_best_models(2)
    score0 = models[0].score(VAL_INPUTS, VAL_TARGETS)
    score1 = models[1].score(VAL_INPUTS, VAL_TARGETS)
    assert score0 >= score1
