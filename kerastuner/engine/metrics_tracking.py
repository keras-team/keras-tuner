# Copyright 2019 The Keras Tuner Authors
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import numpy as np


class MetricsTracker(object):
    def __init__(self, metrics=None):
        self.names = []
        self.directions = {}
        self.metrics_history = {}
        self.register_metrics(metrics)

    def exists(self, name):
        return name in self.names

    def register_metrics(self, metrics=None):
        metrics = metrics or []
        for metric in metrics:
            direction = infer_metric_direction(metric)
            self.register(metric.name, direction)

    def register(self, name, direction=None):
        """Register a metric by name and direction."""

        if direction is None:
            direction = infer_metric_direction_by_name(name)

        if direction not in {'min', 'max'}:
            raise ValueError('`direction` should be one of '
                             '{"min", "max"}, but got: %s' % (direction, ))
        if name in self.names:
            raise ValueError('Metric already exists: %s' % (name, ))
        self.names.append(name)
        self.directions[name] = direction
        self.metrics_history[name] = []

    def update(self, name, value):
        value = float(value)
        if not self.exists(name):
            self.register(name)
        history = self.get_history(name)
        if not history:
            improved = True
        elif self.directions[name] == 'max' and value >= np.max(history):
            improved = True
        elif self.directions[name] == 'min' and value <= np.max(history):
            improved = True
        else:
            improved = False
        history.append(value)
        return improved

    def get_history(self, name):
        if name not in self.names:
            raise ValueError('Unknown metric: %s' % (name, ))
        return self.metrics_history[name]

    def set_history(self, name, series):
        assert type(series) == list
        if not self.exists(name):
            self.register(name)
        self.metrics_history[name] = series

    def get_best_value(self, name):
        history = self.get_history(name)
        direction = self.directions[name]

        if direction == 'min' and len(history):
            return min(history)
        elif direction == 'max' and len(history):
            return max(history)
        else:
            return None

    def get_statistics(self, name):
        history = self.get_history(name)
        if not len(history):
            return {}
        return {
            'min': float(np.min(history)),
            'max': float(np.max(history)),
            'mean': float(np.mean(history)),
            'median': float(np.median(history)),
            'var': float(np.var(history)),
            'std': float(np.std(history))
        }

    def get_last_value(self, name):
        history = self.get_history(name)
        if history:
            return history[-1]
        else:
            return None

    def get_config(self):
        return {
            'names': copy.copy(self.names),
            'directions': copy.copy(self.directions),
            'metrics_history': copy.copy(self.metrics_history)
        }

    @classmethod
    def from_config(cls, config):
        instance = cls()
        instance.names = config['names']
        instance.directions = config['directions']
        instance.metrics_history = config['metrics_history']
        return instance


_MAX_METRICS = {
    'Accuracy', 'BinaryAccuracy', 'CategoricalAccuracy',
    'SparseCategoricalAccuracy', 'TopKCategoricalAccuracy',
    'SparseTopKCategoricalAccuracy', 'TruePositives', 'TrueNegatives',
    'Precision', 'Recall', 'AUC', 'SensitivityAtSpecificity',
    'SpecificityAtSensitivity'
}

_MAX_METRIC_FNS = {
    'accuracy', 'categorical_accuracy', 'binary_accuracy',
    'sparse_categorical_accuracy'
}


def infer_metric_direction_by_name(name):
    """Infer the metric direction based on a name.

    Assumes that if the name exists in _MAX_METRICS or _MAX_METRIC_FNS, the
    metric should be sorted in 'max' order, otherwise 'min'.

    If the name starts with 'val_', said prefix will be ignored.
    """
    if name.startswith("val_"):
        name = name[4:]

    if name in _MAX_METRICS or name in _MAX_METRIC_FNS:
        return "max"
    return "min"


def infer_metric_direction(metric):
    name = metric.__class__.__name__
    direction = 'min'
    if name in _MAX_METRICS:
        direction = 'max'
    elif name == 'MeanMetricWrapper':
        wrapped_fn = metric._fn
        if hasattr(wrapped_fn, '__name__'):
            if wrapped_fn.__name__ in _MAX_METRIC_FNS:
                direction = 'max'
    return direction
