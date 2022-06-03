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

import numpy as np
import pandas as pd
import pytest
from sklearn import datasets
from sklearn import decomposition
from sklearn import ensemble
from sklearn import linear_model
from sklearn import metrics
from sklearn import model_selection
from sklearn import neighbors
from sklearn import pipeline

import keras_tuner


def build_model(hp):
    model_type = hp.Choice("model_type", ["random_forest", "ridge", "knn"])
    if model_type == "random_forest":
        with hp.conditional_scope("model_type", "random_forest"):
            model = ensemble.RandomForestClassifier(
                n_estimators=hp.Int("n_estimators", 10, 50, step=10),
                max_depth=hp.Int("max_depth", 3, 10),
            )
    elif model_type == "ridge":
        with hp.conditional_scope("model_type", "ridge"):
            model = linear_model.RidgeClassifier(
                alpha=hp.Float("alpha", 1e-3, 1, sampling="log")
            )
    elif model_type == "knn":
        with hp.conditional_scope("model_type", "knn"):
            k = hp.Int("n_neighbors", 1, 30, default=5)
            model = neighbors.KNeighborsClassifier(
                n_neighbors=k,
                weights=hp.Choice(
                    "weights", ["uniform", "distance"], default="uniform"
                ),
            )
    else:
        raise ValueError("Unrecognized model_type")
    return model


def build_pipeline(hp):
    n_components = hp.Choice("n_components", [2, 5, 10], default=5)
    pca = decomposition.PCA(n_components=n_components)

    model_type = hp.Choice("model_type", ["random_forest", "ridge", "knn"])
    if model_type == "random_forest":
        with hp.conditional_scope("model_type", "random_forest"):
            model = ensemble.RandomForestClassifier(
                n_estimators=hp.Int("n_estimators", 10, 50, step=10),
                max_depth=hp.Int("max_depth", 3, 10),
            )
    elif model_type == "ridge":
        with hp.conditional_scope("model_type", "ridge"):
            model = linear_model.RidgeClassifier(
                alpha=hp.Float("alpha", 1e-3, 1, sampling="log")
            )
    elif model_type == "knn":
        with hp.conditional_scope("model_type", "knn"):
            k = hp.Int("n_neighbors", 1, 30, default=5)
            model = neighbors.KNeighborsClassifier(
                n_neighbors=k,
                weights=hp.Choice(
                    "weights", ["uniform", "distance"], default="uniform"
                ),
            )
    else:
        raise ValueError("Unrecognized model_type")

    skpipeline = pipeline.Pipeline([("pca", pca), ("clf", model)])
    return skpipeline


def test_sklearn_tuner_simple_with_np(tmp_path):
    tuner = keras_tuner.SklearnTuner(
        oracle=keras_tuner.oracles.BayesianOptimization(
            objective=keras_tuner.Objective("score", "max"), max_trials=10
        ),
        hypermodel=build_model,
        directory=tmp_path,
    )

    x = np.random.uniform(size=(50, 10))
    y = np.random.randint(0, 2, size=(50,))
    tuner.search(x, y)

    assert len(tuner.oracle.trials) == 10

    best_trial = tuner.oracle.get_best_trials()[0]
    assert best_trial.status == "COMPLETED"
    assert best_trial.score is not None
    assert best_trial.best_step == 0
    assert best_trial.metrics.exists("score")

    # Make sure best model can be reloaded.
    best_model = tuner.get_best_models()[0]
    best_model.score(x, y)


@pytest.mark.filterwarnings("ignore:.*column-vector")
def test_sklearn_tuner_with_df(tmp_path):
    tuner = keras_tuner.SklearnTuner(
        oracle=keras_tuner.oracles.BayesianOptimization(
            objective=keras_tuner.Objective("score", "max"), max_trials=10
        ),
        hypermodel=build_model,
        directory=tmp_path,
    )

    x = pd.DataFrame(np.random.uniform(size=(50, 10)))
    y = pd.DataFrame(np.random.randint(0, 2, size=(50,)))
    tuner.search(x, y)

    assert len(tuner.oracle.trials) == 10


def test_sklearn_custom_scoring_and_cv(tmp_path):
    tuner = keras_tuner.SklearnTuner(
        oracle=keras_tuner.oracles.BayesianOptimization(
            objective=keras_tuner.Objective("score", "max"), max_trials=10
        ),
        hypermodel=build_model,
        scoring=metrics.make_scorer(metrics.balanced_accuracy_score),
        cv=model_selection.StratifiedKFold(5),
        directory=tmp_path,
    )

    x = np.random.uniform(size=(50, 10))
    y = np.random.randint(0, 2, size=(50,))
    tuner.search(x, y)

    assert len(tuner.oracle.trials) == 10

    best_trial = tuner.oracle.get_best_trials()[0]
    assert best_trial.status == "COMPLETED"
    assert best_trial.score is not None
    assert best_trial.best_step == 0
    assert best_trial.metrics.exists("score")

    # Make sure best model can be reloaded.
    best_model = tuner.get_best_models()[0]
    best_model.score(x, y)


def test_sklearn_additional_metrics(tmp_path):
    tuner = keras_tuner.SklearnTuner(
        oracle=keras_tuner.oracles.BayesianOptimization(
            objective=keras_tuner.Objective("score", "max"), max_trials=10
        ),
        hypermodel=build_model,
        metrics=[metrics.balanced_accuracy_score, metrics.recall_score],
        directory=tmp_path,
    )

    x = np.random.uniform(size=(50, 10))
    y = np.random.randint(0, 2, size=(50,))
    tuner.search(x, y)

    assert len(tuner.oracle.trials) == 10

    best_trial = tuner.oracle.get_best_trials()[0]
    assert best_trial.status == "COMPLETED"
    assert best_trial.score is not None
    assert best_trial.best_step == 0
    assert best_trial.metrics.exists("score")
    assert best_trial.metrics.exists("balanced_accuracy_score")
    assert best_trial.metrics.exists("recall_score")

    # Make sure best model can be reloaded.
    best_model = tuner.get_best_models()[0]
    best_model.score(x, y)


def test_sklearn_sample_weight(tmp_path):
    tuner = keras_tuner.SklearnTuner(
        oracle=keras_tuner.oracles.BayesianOptimization(
            objective=keras_tuner.Objective("score", "max"), max_trials=10
        ),
        hypermodel=build_model,
        directory=tmp_path,
    )

    x = np.random.uniform(size=(50, 10))
    y = np.random.randint(0, 2, size=(50,))
    sample_weight = np.random.uniform(0.1, 1, size=(50,))
    tuner.search(x, y, sample_weight=sample_weight)

    assert len(tuner.oracle.trials) == 10

    best_trial = tuner.oracle.get_best_trials()[0]
    assert best_trial.status == "COMPLETED"
    assert best_trial.score is not None
    assert best_trial.best_step == 0
    assert best_trial.metrics.exists("score")

    # Make sure best model can be reloaded.
    best_model = tuner.get_best_models()[0]
    best_model.score(x, y)


def test_sklearn_pipeline(tmp_path):
    tuner = keras_tuner.SklearnTuner(
        oracle=keras_tuner.oracles.BayesianOptimization(
            objective=keras_tuner.Objective("score", "max"), max_trials=10
        ),
        hypermodel=build_pipeline,
        directory=tmp_path,
    )

    x = np.random.uniform(size=(50, 10))
    y = np.random.randint(0, 2, size=(50,))
    sample_weight = np.random.uniform(0.1, 1, size=(50,))
    tuner.search(x, y, sample_weight=sample_weight)

    assert len(tuner.oracle.trials) == 10

    best_trial = tuner.oracle.get_best_trials()[0]
    assert best_trial.status == "COMPLETED"
    assert best_trial.score is not None
    assert best_trial.best_step == 0
    assert best_trial.metrics.exists("score")

    # Make sure best pipeline can be reloaded.
    best_pipeline = tuner.get_best_models()[0]
    best_pipeline.score(x, y)


def test_sklearn_cv_with_groups(tmp_path):
    tuner = keras_tuner.SklearnTuner(
        oracle=keras_tuner.oracles.BayesianOptimization(
            objective=keras_tuner.Objective("score", "max"), max_trials=10
        ),
        hypermodel=build_model,
        cv=model_selection.GroupKFold(5),
        directory=tmp_path,
    )

    x = np.random.uniform(size=(50, 10))
    y = np.random.randint(0, 2, size=(50,))
    groups = np.random.randint(0, 5, size=(50,))
    tuner.search(x, y, groups=groups)

    assert len(tuner.oracle.trials) == 10

    best_trial = tuner.oracle.get_best_trials()[0]
    assert best_trial.status == "COMPLETED"
    assert best_trial.score is not None
    assert best_trial.best_step == 0
    assert best_trial.metrics.exists("score")

    # Make sure best model can be reloaded.
    best_model = tuner.get_best_models()[0]
    best_model.score(x, y)


def test_sklearn_real_data(tmp_path):
    tuner = keras_tuner.SklearnTuner(
        oracle=keras_tuner.oracles.BayesianOptimization(
            objective=keras_tuner.Objective("score", "max"), max_trials=10
        ),
        hypermodel=build_model,
        scoring=metrics.make_scorer(metrics.accuracy_score),
        cv=model_selection.StratifiedKFold(5),
        directory=tmp_path,
    )

    x, y = datasets.load_iris(return_X_y=True)
    x_train, x_test, y_train, y_test = model_selection.train_test_split(
        x, y, test_size=0.2
    )

    tuner.search(x_train, y_train)

    best_models = tuner.get_best_models(10)
    best_model = best_models[0]
    worst_model = best_models[9]
    best_model_score = best_model.score(x_test, y_test)
    worst_model_score = worst_model.score(x_test, y_test)

    assert best_model_score > 0.8
    assert best_model_score >= worst_model_score


def test_sklearn_not_install_error(tmp_path):
    sklearn_module = keras_tuner.tuners.sklearn_tuner.sklearn
    keras_tuner.tuners.sklearn_tuner.sklearn = None

    with pytest.raises(ImportError, match="Please install sklearn"):
        keras_tuner.SklearnTuner(
            oracle=keras_tuner.oracles.BayesianOptimization(
                objective=keras_tuner.Objective("score", "max"), max_trials=10
            ),
            hypermodel=build_model,
            directory=tmp_path,
        )

    keras_tuner.tuners.sklearn_tuner.sklearn = sklearn_module


def test_sklearn_deprecation_warning(tmp_path):
    with pytest.deprecated_call():
        keras_tuner.tuners.Sklearn(
            oracle=keras_tuner.oracles.BayesianOptimization(
                objective=keras_tuner.Objective("score", "max"), max_trials=10
            ),
            hypermodel=build_model,
            directory=tmp_path,
        )
