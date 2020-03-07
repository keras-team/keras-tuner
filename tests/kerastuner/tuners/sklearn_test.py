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
import pytest
from sklearn import datasets
from sklearn import ensemble
from sklearn import linear_model
from sklearn import metrics
from sklearn import model_selection

import kerastuner as kt
from kerastuner.tuners import sklearn_tuner


def build_model(hp):
    model_type = hp.Choice('model_type', ['random_forest', 'ridge'])
    if model_type == 'random_forest':
        with hp.conditional_scope('model_type', 'random_forest'):
            model = ensemble.RandomForestClassifier(
                n_estimators=hp.Int('n_estimators', 10, 50, step=10),
                max_depth=hp.Int('max_depth', 3, 10))
    elif model_type == 'ridge':
        with hp.conditional_scope('model_type', 'ridge'):
            model = linear_model.RidgeClassifier(
                alpha=hp.Float('alpha', 1e-3, 1, sampling='log'))
    else:
        raise ValueError('Unrecognized model_type')
    return model


@pytest.fixture(scope='function')
def tmp_dir(tmpdir_factory):
    return tmpdir_factory.mktemp('hyperband_test', numbered=True)


def test_sklearn_tuner_simple(tmp_dir):
    tuner = sklearn_tuner.Sklearn(
        oracle=kt.oracles.BayesianOptimization(
            objective=kt.Objective('score', 'max'),
            max_trials=10),
        hypermodel=build_model,
        directory=tmp_dir)

    x = np.random.uniform(size=(50, 10))
    y = np.random.randint(0, 2, size=(50,))
    tuner.search(x, y)

    assert len(tuner.oracle.trials) == 10

    best_trial = tuner.oracle.get_best_trials()[0]
    assert best_trial.status == 'COMPLETED'
    assert best_trial.score is not None
    assert best_trial.best_step == 0
    assert best_trial.metrics.exists('score')

    # Make sure best model can be reloaded.
    best_model = tuner.get_best_models()[0]
    best_model.score(x, y)


def test_sklearn_custom_scoring_and_cv(tmp_dir):
    tuner = sklearn_tuner.Sklearn(
        oracle=kt.oracles.BayesianOptimization(
            objective=kt.Objective('score', 'max'),
            max_trials=10),
        hypermodel=build_model,
        scoring=metrics.make_scorer(metrics.balanced_accuracy_score),
        cv=model_selection.StratifiedKFold(5),
        directory=tmp_dir)

    x = np.random.uniform(size=(50, 10))
    y = np.random.randint(0, 2, size=(50,))
    tuner.search(x, y)

    assert len(tuner.oracle.trials) == 10

    best_trial = tuner.oracle.get_best_trials()[0]
    assert best_trial.status == 'COMPLETED'
    assert best_trial.score is not None
    assert best_trial.best_step == 0
    assert best_trial.metrics.exists('score')

    # Make sure best model can be reloaded.
    best_model = tuner.get_best_models()[0]
    best_model.score(x, y)


def test_sklearn_additional_metrics(tmp_dir):
    tuner = sklearn_tuner.Sklearn(
        oracle=kt.oracles.BayesianOptimization(
            objective=kt.Objective('score', 'max'),
            max_trials=10),
        hypermodel=build_model,
        metrics=[metrics.balanced_accuracy_score,
                 metrics.recall_score],
        directory=tmp_dir)

    x = np.random.uniform(size=(50, 10))
    y = np.random.randint(0, 2, size=(50,))
    tuner.search(x, y)

    assert len(tuner.oracle.trials) == 10

    best_trial = tuner.oracle.get_best_trials()[0]
    assert best_trial.status == 'COMPLETED'
    assert best_trial.score is not None
    assert best_trial.best_step == 0
    assert best_trial.metrics.exists('score')
    assert best_trial.metrics.exists('balanced_accuracy_score')
    assert best_trial.metrics.exists('recall_score')

    # Make sure best model can be reloaded.
    best_model = tuner.get_best_models()[0]
    best_model.score(x, y)


def test_sklearn_sample_weight(tmp_dir):
    tuner = sklearn_tuner.Sklearn(
        oracle=kt.oracles.BayesianOptimization(
            objective=kt.Objective('score', 'max'),
            max_trials=10),
        hypermodel=build_model,
        directory=tmp_dir)

    x = np.random.uniform(size=(50, 10))
    y = np.random.randint(0, 2, size=(50,))
    sample_weight = np.random.uniform(0.1, 1, size=(50,))
    tuner.search(x, y, sample_weight=sample_weight)

    assert len(tuner.oracle.trials) == 10

    best_trial = tuner.oracle.get_best_trials()[0]
    assert best_trial.status == 'COMPLETED'
    assert best_trial.score is not None
    assert best_trial.best_step == 0
    assert best_trial.metrics.exists('score')

    # Make sure best model can be reloaded.
    best_model = tuner.get_best_models()[0]
    best_model.score(x, y)


def test_sklearn_cv_with_groups(tmp_dir):
    tuner = sklearn_tuner.Sklearn(
        oracle=kt.oracles.BayesianOptimization(
            objective=kt.Objective('score', 'max'),
            max_trials=10),
        hypermodel=build_model,
        cv=model_selection.GroupKFold(5),
        directory=tmp_dir)

    x = np.random.uniform(size=(50, 10))
    y = np.random.randint(0, 2, size=(50,))
    groups = np.random.randint(0, 5, size=(50,))
    tuner.search(x, y, groups=groups)

    assert len(tuner.oracle.trials) == 10

    best_trial = tuner.oracle.get_best_trials()[0]
    assert best_trial.status == 'COMPLETED'
    assert best_trial.score is not None
    assert best_trial.best_step == 0
    assert best_trial.metrics.exists('score')

    # Make sure best model can be reloaded.
    best_model = tuner.get_best_models()[0]
    best_model.score(x, y)


def test_sklearn_real_data(tmp_dir):
    tuner = sklearn_tuner.Sklearn(
        oracle=kt.oracles.BayesianOptimization(
            objective=kt.Objective('score', 'max'),
            max_trials=10),
        hypermodel=build_model,
        scoring=metrics.make_scorer(metrics.accuracy_score),
        cv=model_selection.StratifiedKFold(5),
        directory=tmp_dir)

    x, y = datasets.load_iris(return_X_y=True)
    x_train, x_test, y_train, y_test = model_selection.train_test_split(
        x, y, test_size=0.2)

    tuner.search(x_train, y_train)

    best_models = tuner.get_best_models(10)
    best_model = best_models[0]
    worst_model = best_models[9]
    best_model_score = best_model.score(x_test, y_test)
    worst_model_score = worst_model.score(x_test, y_test)

    assert best_model_score > 0.8
    assert best_model_score >= worst_model_score
