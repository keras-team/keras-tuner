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
from tensorflow.keras import losses
from tensorflow.keras import metrics


def test_register_from_metrics():
    # As well as direction inference.
    tracker = metrics_tracking.MetricsTracker(
        metrics=[metrics.CategoricalAccuracy(),
                 metrics.MeanSquaredError()]
    )
    assert set(tracker.metrics.keys()) == {'categorical_accuracy',
                                           'mean_squared_error'}
    assert tracker.metrics['categorical_accuracy'].direction == 'max'
    assert tracker.metrics['mean_squared_error'].direction == 'min'


def test_register():
    tracker = metrics_tracking.MetricsTracker()
    tracker.register('new_metric', direction='max')
    assert set(tracker.metrics.keys()) == {'new_metric'}
    assert tracker.metrics['new_metric'].direction == 'max'
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
    assert set(tracker.metrics.keys()) == {'new_metric'}
    assert tracker.metrics['new_metric'].direction == 'min'  # default direction
    assert (tracker.get_history('new_metric') ==
            [metrics_tracking.MetricObservation(0.5, step=0)])


def test_get_history():
    tracker = metrics_tracking.MetricsTracker()
    tracker.update('new_metric', 0.5, step=0)
    tracker.update('new_metric', 1.5, step=1)
    tracker.update('new_metric', 2., step=2)
    assert tracker.get_history('new_metric') == [
        metrics_tracking.MetricObservation(0.5, 0),
        metrics_tracking.MetricObservation(1.5, 1),
        metrics_tracking.MetricObservation(2., 2),
    ]
    with pytest.raises(ValueError,  match='Unknown metric'):
        tracker.get_history('another_metric')


def test_set_history():
    tracker = metrics_tracking.MetricsTracker()
    tracker.set_history('new_metric', [
        metrics_tracking.MetricObservation(0.5, 0),
        metrics_tracking.MetricObservation(1.5, 1),
        metrics_tracking.MetricObservation(2., 2),
    ])
    values = [obs.value for obs in tracker.get_history('new_metric')]
    steps = [obs.step for obs in tracker.get_history('new_metric')]
    assert values == [[0.5], [1.5], [2.]]
    assert steps == [0, 1, 2]


def test_get_best_value():
    tracker = metrics_tracking.MetricsTracker()
    tracker.register('metric_min', 'min')
    tracker.register('metric_max', 'max')
    assert tracker.get_best_value('metric_min') is None

    tracker.set_history(
        'metric_min',
        [metrics_tracking.MetricObservation(1., 0),
         metrics_tracking.MetricObservation(2., 1),
         metrics_tracking.MetricObservation(3., 2)])
    tracker.set_history(
        'metric_max',
        [metrics_tracking.MetricObservation(1., 0),
         metrics_tracking.MetricObservation(2., 1),
         metrics_tracking.MetricObservation(3., 2)])
    assert tracker.get_best_value('metric_min') == 1.
    assert tracker.get_best_value('metric_max') == 3.


def test_get_statistics():
    tracker = metrics_tracking.MetricsTracker()
    history = [
        metrics_tracking.MetricObservation(random.random(), i)
        for i in range(10)]
    tracker.set_history('new_metric', history)
    stats = tracker.get_statistics('new_metric')
    assert set(stats.keys()) == {
        'min', 'max', 'mean', 'median', 'var', 'std'}
    history = [obs.value for obs in history]
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
    tracker.set_history(
        'new_metric',
        [metrics_tracking.MetricObservation(1., 0),
         metrics_tracking.MetricObservation(2., 1),
         metrics_tracking.MetricObservation(3., 2)])
    assert tracker.get_last_value('new_metric') == 3.


def test_serialization():
    tracker = metrics_tracking.MetricsTracker()
    tracker.register('metric_min', 'min')
    tracker.register('metric_max', 'max')
    tracker.set_history(
        'metric_min',
        [metrics_tracking.MetricObservation(1., 0),
         metrics_tracking.MetricObservation(2., 1),
         metrics_tracking.MetricObservation(3., 2)])
    tracker.set_history(
        'metric_max',
        [metrics_tracking.MetricObservation(1., 0),
         metrics_tracking.MetricObservation(2., 1),
         metrics_tracking.MetricObservation(3., 2)])

    new_tracker = metrics_tracking.MetricsTracker.from_config(
        tracker.get_config())
    assert new_tracker.metrics.keys() == tracker.metrics.keys()


def test_metricobservation_proto():
    obs = metrics_tracking.MetricObservation(-10, 5)
    proto = obs.to_proto()
    assert proto.value == [-10]
    assert proto.step == 5
    new_obs = metrics_tracking.MetricObservation.from_proto(proto)
    assert new_obs == obs


def test_metrichistory_proto():
    tracker = metrics_tracking.MetricHistory('max')
    tracker.update(5, step=3)
    tracker.update(10, step=4)

    proto = tracker.to_proto()
    assert proto.maximize
    assert proto.observations[0].value == [5]
    assert proto.observations[0].step == 3
    assert proto.observations[1].value == [10]
    assert proto.observations[1].step == 4

    new_tracker = metrics_tracking.MetricHistory.from_proto(proto)
    assert new_tracker.direction == 'max'
    assert new_tracker.get_history() == [
        metrics_tracking.MetricObservation(5, 3),
        metrics_tracking.MetricObservation(10, 4)]


def test_metricstracker_proto():
    tracker = metrics_tracking.MetricsTracker()
    tracker.register('score', direction='max')
    tracker.update('score', value=10, step=1)
    tracker.update('score', value=20, step=1)
    tracker.update('score', value=30, step=2)

    proto = tracker.to_proto()
    obs = proto.metrics['score'].observations
    assert obs[0].value == [10, 20]
    assert obs[0].step == 1
    assert obs[1].value == [30]
    assert obs[1].step == 2
    assert proto.metrics['score'].maximize

    new_tracker = metrics_tracking.MetricsTracker.from_proto(proto)
    assert new_tracker.metrics['score'].direction == 'max'
    assert new_tracker.metrics['score'].get_history() == [
        metrics_tracking.MetricObservation([10, 20], 1),
        metrics_tracking.MetricObservation(30, 2)]


def test_metric_direction_inference():
    # Test min metrics.
    assert metrics_tracking.infer_metric_direction('MAE') == 'min'
    assert metrics_tracking.infer_metric_direction(
        metrics.binary_crossentropy) == 'min'
    assert metrics_tracking.infer_metric_direction(
        metrics.FalsePositives()) == 'min'

    # All losses in keras.losses are considered as 'min'.
    assert metrics_tracking.infer_metric_direction(
        'squared_hinge') == 'min'
    assert metrics_tracking.infer_metric_direction(
        losses.hinge) == 'min'
    assert metrics_tracking.infer_metric_direction(
        losses.CategoricalCrossentropy()) == 'min'

    # Test max metrics.
    assert metrics_tracking.infer_metric_direction(
        'binary_accuracy') == 'max'
    assert metrics_tracking.infer_metric_direction(
        metrics.categorical_accuracy) == 'max'
    assert metrics_tracking.infer_metric_direction(
        metrics.Precision()) == 'max'

    # Test unknown metrics.
    assert metrics_tracking.infer_metric_direction('my_metric') is None

    def my_metric_fn(x, y):
        return x
    assert metrics_tracking.infer_metric_direction(my_metric_fn) is None

    class MyMetric(metrics.Metric):
        def update_state(self, x, y):
            return 1

        def result(self):
            return 1
    assert metrics_tracking.infer_metric_direction(MyMetric()) is None

    # Test special cases.
    assert metrics_tracking.infer_metric_direction('loss') == 'min'
    assert metrics_tracking.infer_metric_direction('acc') == 'max'
    assert metrics_tracking.infer_metric_direction('val_acc') == 'max'
    assert metrics_tracking.infer_metric_direction('crossentropy') == 'min'
    assert metrics_tracking.infer_metric_direction('ce') == 'min'
    assert metrics_tracking.infer_metric_direction('weighted_acc') == 'max'
    assert metrics_tracking.infer_metric_direction('val_weighted_ce') == 'min'
    assert metrics_tracking.infer_metric_direction(
        'weighted_binary_accuracy') == 'max'
