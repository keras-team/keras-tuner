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
import random
import numpy as np

from kerastuner.engine import metrics_tracking
from tensorflow.keras import metrics


def test_register_from_metrics():
    # As well as direction inference.
    tracker = metrics_tracking.MetricsTracker(
        metrics=[metrics.CategoricalAccuracy(),
                 metrics.MeanSquaredError()]
    )
    assert tracker.names == ['categorical_accuracy',
                             'mean_squared_error']
    assert tracker.directions['categorical_accuracy'] == 'max'
    assert tracker.directions['mean_squared_error'] == 'min'
    # TODO: better coverage for direction inference.


def test_register():
    tracker = metrics_tracking.MetricsTracker()
    tracker.register('new_metric', direction='max')
    assert tracker.names == ['new_metric']
    assert tracker.directions['new_metric'] == 'max'
    with pytest.raises(ValueError,
                       match='`direction` should be one of'):
        tracker.register('another_metric', direction='wrong')
    with pytest.raises(ValueError,
                       match='already exists'):
        tracker.register('new_metric', direction='max')


def test_exists():
    tracker = metrics_tracking.MetricsTracker()
    tracker.register('new_metric', direction='max')
    assert tracker.exists('new_metric')
    assert not tracker.exists('another_metric')


def test_update():
    tracker = metrics_tracking.MetricsTracker()
    tracker.update('new_metric', 0.5)  # automatic registration
    assert tracker.names == ['new_metric']
    assert tracker.directions['new_metric'] == 'min'  # default direction
    assert tracker.get_history('new_metric') == [0.5]


def test_get_history():
    tracker = metrics_tracking.MetricsTracker()
    tracker.update('new_metric', 0.5)
    tracker.update('new_metric', 1.5)
    tracker.update('new_metric', 2.)
    assert tracker.get_history('new_metric') == [0.5, 1.5, 2.]
    with pytest.raises(ValueError,  match='Unknown metric'):
        tracker.get_history('another_metric')


def test_set_history():
    tracker = metrics_tracking.MetricsTracker()
    tracker.set_history('new_metric', [1., 2., 3.])
    tracker.get_history('new_metric') == [1., 2., 3.]


def test_get_best_value():
    tracker = metrics_tracking.MetricsTracker()
    tracker.register('metric_min', 'min')
    tracker.register('metric_max', 'max')
    assert tracker.get_best_value('metric_min') is None

    tracker.set_history('metric_min', [1., 2., 3.])
    tracker.set_history('metric_max', [1., 2., 3.])
    assert tracker.get_best_value('metric_min') == 1.
    assert tracker.get_best_value('metric_max') == 3.


def test_get_statistics():
    tracker = metrics_tracking.MetricsTracker()
    history = [random.random() for _ in range(10)]
    tracker.set_history('new_metric', history)
    stats = tracker.get_statistics('new_metric')
    assert set(stats.keys()) == {
        'min', 'max', 'mean', 'median', 'var', 'std'}
    assert stats['min'] == np.min(history)
    assert stats['max'] == np.max(history)
    assert stats['mean'] == np.mean(history)
    assert stats['median'] == np.median(history)
    assert stats['var'] == np.var(history)
    assert stats['std'] == np.std(history)


def test_get_last_value():
    tracker = metrics_tracking.MetricsTracker()
    tracker.register('new_metric', 'min')
    assert tracker.get_last_value('new_metric') is None
    tracker.set_history('new_metric', [1., 2., 3.])
    assert tracker.get_last_value('new_metric') is 3.


def test_serialization():
    tracker = metrics_tracking.MetricsTracker()
    tracker.register('metric_min', 'min')
    tracker.register('metric_max', 'max')
    tracker.set_history('metric_min', [1., 2., 3.])
    tracker.set_history('metric_max', [1., 2., 3.])

    new_tracker = metrics_tracking.MetricsTracker.from_config(
        tracker.get_config())
    assert new_tracker.names == tracker.names
    assert new_tracker.directions == tracker.directions
    assert new_tracker.metrics_history == tracker.metrics_history
