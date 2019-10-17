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
import math
import numpy as np
import random
import sklearn
import tensorflow as tf

from ..engine import base_tuner

class Sklearn(base_tuner.BaseTuner):
    """Tuner for Scikit-learn Models.

    Performs cross-validated hyperparameter searching.
    """
    def __init__(self,
                 oracle,
                 hypermodel,
                 scoring=None,
                 metrics=None,
                 cross_validation=sklearn.model_selection.KFold(5, shuffle=True),
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

            scoring = model.score if self.scoring is None else self.scoring
            metrics['score'].append(scoring(X_test,
                                            y_test,
                                            sample_weight=sample_weight_test))
            for metric in self.metrics:
                metrics[metric.__name__].append(metric(X_test,
                                                       y_test,
                                                       sample_weight=sample_weight_test))

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
