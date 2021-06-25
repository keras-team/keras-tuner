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
"""Tuner for Scikit-learn Models."""
import collections
import os
import pickle
import warnings

import numpy as np
import pandas as pd
import tensorflow as tf

try:
    import sklearn  # pytype: disable=import-error
except ImportError:
    sklearn = None

from ..engine import base_tuner


def split_data(data, indices):
    if isinstance(data, np.ndarray):
        return data[indices]
    elif isinstance(data, pd.DataFrame):
        return data.iloc[indices]
    else:
        raise TypeError()


class SklearnTuner(base_tuner.BaseTuner):
    """Tuner for Scikit-learn Models.

    Performs cross-validated hyperparameter search for Scikit-learn models.

    Examples:

    ```python
    import keras_tuner as kt
    from sklearn import ensemble
    from sklearn import datasets
    from sklearn import linear_model
    from sklearn import metrics
    from sklearn import model_selection

    def build_model(hp):
      model_type = hp.Choice('model_type', ['random_forest', 'ridge'])
      if model_type == 'random_forest':
        model = ensemble.RandomForestClassifier(
            n_estimators=hp.Int('n_estimators', 10, 50, step=10),
            max_depth=hp.Int('max_depth', 3, 10))
      else:
        model = linear_model.RidgeClassifier(
            alpha=hp.Float('alpha', 1e-3, 1, sampling='log'))
      return model

    tuner = kt.tuners.SklearnTuner(
        oracle=kt.oracles.BayesianOptimizationOracle(
            objective=kt.Objective('score', 'max'),
            max_trials=10),
        hypermodel=build_model,
        scoring=metrics.make_scorer(metrics.accuracy_score),
        cv=model_selection.StratifiedKFold(5),
        directory='.',
        project_name='my_project')

    X, y = datasets.load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        X, y, test_size=0.2)

    tuner.search(X_train, y_train)

    best_model = tuner.get_best_models(num_models=1)[0]
    ```

    Args:
        oracle: A `keras_tuner.Oracle` instance. Note that for this `Tuner`,
            the `objective` for the `Oracle` should always be set to
            `Objective('score', direction='max')`. Also, `Oracle`s that exploit
            Neural-Network-specific training (e.g. `Hyperband`) should not be
            used with this `Tuner`.
        hypermodel: A `HyperModel` instance (or callable that takes
            hyperparameters and returns a Model instance).
        scoring: An sklearn `scoring` function. For more information, see
            `sklearn.metrics.make_scorer`. If not provided, the Model's default
            scoring will be used via `model.score`. Note that if you are
            searching across different Model families, the default scoring for
            these Models will often be different. In this case you should
            supply `scoring` here in order to make sure your Models are being
            scored on the same metric.
        metrics: Additional `sklearn.metrics` functions to monitor during search.
            Note that these metrics do not affect the search process.
        cv: An `sklearn.model_selection` Splitter class. Used to
            determine how samples are split up into groups for
            cross-validation.
        **kwargs: Keyword arguments relevant to all `Tuner` subclasses. Please
            see the docstring for `Tuner`.
    """

    def __init__(
        self, oracle, hypermodel, scoring=None, metrics=None, cv=None, **kwargs
    ):
        super().__init__(oracle=oracle, hypermodel=hypermodel, **kwargs)

        if sklearn is None:
            raise ImportError(
                "Please install sklearn before using the `SklearnTuner`."
            )

        self.scoring = scoring

        if metrics is None:
            metrics = []
        if not isinstance(metrics, (list, tuple)):
            metrics = [metrics]
        self.metrics = metrics

        self.cv = cv or sklearn.model_selection.KFold(
            5, shuffle=True, random_state=1
        )

    def search(self, X, y, sample_weight=None, groups=None):
        """Performs hyperparameter search.

        Args:
            X: See docstring for `model.fit` for the `sklearn` Models being tuned.
            y: See docstring for `model.fit` for the `sklearn` Models being tuned.
            sample_weight: Optional. See docstring for `model.fit` for the
                `sklearn` Models being tuned.
            groups: Optional. Required for `sklearn.model_selection` Splitter
                classes that split based on group labels (For example, see
                `sklearn.model_selection.GroupKFold`).
        """
        # Only overridden for the docstring.
        return super().search(X, y, sample_weight=sample_weight, groups=groups)

    def run_trial(self, trial, X, y, sample_weight=None, groups=None):

        metrics = collections.defaultdict(list)
        # For cross-validation methods that expect a `groups` argument.
        cv_kwargs = {"groups": groups} if groups is not None else {}
        for train_indices, test_indices in self.cv.split(X, y, **cv_kwargs):
            X_train = split_data(X, train_indices)
            y_train = split_data(y, train_indices)
            X_test = split_data(X, test_indices)
            y_test = split_data(y, test_indices)

            sample_weight_train = (
                sample_weight[train_indices] if sample_weight is not None else None
            )

            model = self.hypermodel.build(trial.hyperparameters)
            if isinstance(model, sklearn.pipeline.Pipeline):
                model.fit(X_train, y_train)
            else:
                model.fit(X_train, y_train, sample_weight=sample_weight_train)

            sample_weight_test = (
                sample_weight[test_indices] if sample_weight is not None else None
            )

            if self.scoring is None:
                score = model.score(X_test, y_test, sample_weight=sample_weight_test)
            else:
                score = self.scoring(
                    model, X_test, y_test, sample_weight=sample_weight_test
                )
            metrics["score"].append(score)

            if self.metrics:
                y_test_pred = model.predict(X_test)
                for metric in self.metrics:
                    result = metric(
                        y_test, y_test_pred, sample_weight=sample_weight_test
                    )
                    metrics[metric.__name__].append(result)

        trial_metrics = {name: np.mean(values) for name, values in metrics.items()}
        self.oracle.update_trial(trial.trial_id, trial_metrics)
        self.save_model(trial.trial_id, model)

    def save_model(self, trial_id, model, step=0):
        fname = os.path.join(self.get_trial_dir(trial_id), "model.pickle")
        with tf.io.gfile.GFile(fname, "wb") as f:
            pickle.dump(model, f)

    def load_model(self, trial):
        fname = os.path.join(self.get_trial_dir(trial.trial_id), "model.pickle")
        with tf.io.gfile.GFile(fname, "rb") as f:
            return pickle.load(f)


class Sklearn(SklearnTuner):
    def __init__(self, *args, **kwargs):
        warnings.warn(
            "The `Sklearn` class is deprecated, please use `SklearnTuner`.",
            DeprecationWarning,
        )
        super().__init__(*args, **kwargs)
