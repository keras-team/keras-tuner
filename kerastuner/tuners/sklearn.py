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
import numpy as np
import os
import pickle
from sklearn import model_selection
import tensorflow as tf

from ..engine import base_tuner


class Sklearn(base_tuner.BaseTuner):
    """Tuner for Scikit-learn Models.

    Performs cross-validated hyperparameter search for Scikit-learn
    models.

    Attributes:
      oracle: An instance of the `kerastuner.Oracle` class. Note that for
        this `Tuner`, the `objective` for the `Oracle` should always be set
        to `Objective('score', direction='max')`. Also, `Oracle`s that exploit
        Neural-Network-specific training (e.g. `Hyperband`) should not be
        used with this `Tuner`.
      hypermodel: Instance of `HyperModel` class (or callable that takes a
        `Hyperparameters` object and returns a Model instance).
      scoring: An sklearn `scoring` function. For more information, see
        `sklearn.metrics.make_scorer`. If not provided, the Model's default
        scoring will be used via `model.score`. Note that if you are searching
        across different Model families, the default scoring for these Models
        will often be different. In this case you should supply `scoring` here
        in order to make sure your Models are being scored on the same metric.
      metrics: Additional `sklearn.metrics` functions to monitor during search.
        Note that these metrics do not affect the search process.
      cross_validation: An `sklearn.model_selection` Splitter class. Used to
        determine how samples are split up into groups for cross-validation.
      **kwargs: Keyword arguments relevant to all `Tuner` subclasses. Please
        see the docstring for `Tuner`.


    Example:

    ```
    import kerastuner as kt
    from sklearn import ensemble
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

    tuner = kt.tuners.Sklearn(
        oracle=kt.oracles.BayesianOptimization(
            objective=kt.Objective('score', 'max'),
            max_trials=10),
        hypermodel=build_model,
        scoring=metrics.make_scorer(metrics.accuracy_score),
        cross_validation=model_selection.StratifiedKFold(5),
        directory='.',
        project_name='my_project')

    X, y = datasets.load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        X, y, test_size=0.2)

    tuner.search(X_train, y_train)

    best_model = tuner.get_best_models(num_models=1)[0]
    ````
    """
    def __init__(self,
                 oracle,
                 hypermodel,
                 scoring=None,
                 metrics=None,
                 cross_validation=model_selection.KFold(5, shuffle=True),
                 **kwargs):
        super(Sklearn, self).__init__(
            oracle=oracle,
            hypermodel=hypermodel,
            **kwargs)

        self.scoring = scoring

        if metrics is None:
            metrics = []
        if not isinstance(metrics, (list, tuple)):
            metrics = [metrics]
        self.metrics = metrics

        self.cross_validation = cross_validation

    def run_trial(self,
                  trial,
                  X,
                  y,
                  sample_weight=None,
                  groups=None):

        metrics = collections.defaultdict(list)
        # For cross-validation methods that expect a `groups` argument.
        cv_kwargs = {'groups': groups} if groups is not None else {}
        for train_indices, test_indices in self.cross_validation.split(
                X, y, **cv_kwargs):
            X_train = X[train_indices]
            y_train = y[train_indices]
            sample_weight_train = (
                sample_weight[train_indices] if sample_weight is not None else None)

            model = self.hypermodel.build(trial.hyperparameters)
            model.fit(X_train, y_train, sample_weight=sample_weight_train)

            X_test = X[test_indices]
            y_test = y[test_indices]
            sample_weight_test = (
                sample_weight[test_indices] if sample_weight is not None else None)

            if self.scoring is None:
                score = model.score(
                    X_test, y_test, sample_weight=sample_weight_test)
            else:
                score = self.scoring(
                    model, X_test, y_test, sample_weight=sample_weight_test)
            metrics['score'].append(score)

            if self.metrics:
                y_test_pred = model.predict(X_test)
                for metric in self.metrics:
                    result = metric(
                        y_test, y_test_pred, sample_weight=sample_weight_test)
                    metrics[metric.__name__].append(result)

        trial_metrics = {name: np.mean(values) for name, values in metrics.items()}
        self.oracle.update_trial(trial.trial_id, trial_metrics)
        self.save_model(trial.trial_id, model)

    def save_model(self, trial_id, model, step=0):
        fname = os.path.join(self.get_trial_dir(trial_id), 'model.pickle')
        with tf.io.gfile.GFile(fname, 'wb') as f:
            pickle.dump(model, f)

    def load_model(self, trial):
        fname = os.path.join(
            self.get_trial_dir(trial.trial_id), 'model.pickle')
        with tf.io.gfile.GFile(fname, 'rb') as f:
            return pickle.load(f)
