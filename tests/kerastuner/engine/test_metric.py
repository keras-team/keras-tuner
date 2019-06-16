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

import json

import numpy as np
import pytest
import sklearn
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model
import time

from sklearn.utils.multiclass import type_of_target
from kerastuner.engine.metric import Metric
from kerastuner.engine.metric import compute_common_classification_metrics


@pytest.fixture
def mm():
    mm = Metric('name', 'min')
    mm.update(10)
    mm.update(11)
    return mm


def test_metric_wall_time():
    mm = Metric('acc', 'max')
    mm.update(10)
    time.sleep(1)
    mm.update(11)
    assert mm.wall_time[1] > 1


def test_metric_creation():
    metric = Metric('test', 'min')
    assert metric.name == 'test'
    assert metric.direction == 'min'
    assert int(metric.start_time) == int(time.time())
    assert metric.wall_time == []
    assert metric.history == []


def test_metric_invalid_direction():
    with pytest.raises(ValueError):
        Metric('test', 'invalid')


def test_best_min_value(mm):
    assert mm.get_best_value() == 10


def test_best_max_value():
    mm = Metric('max', 'max')
    mm.update(10)
    mm.update(8)
    assert mm.get_best_value() == 10


def test_last_value(mm):
    assert mm.get_last_value() == 11


def test_get_empty_last_value():
    mm = Metric('min', 'min')
    assert not mm.get_last_value()


def test_update_improve(mm):
    assert mm.update(6)


def test_update_dont_improve(mm):
    assert not mm.update(3713)


def test_single_update():
    mm = Metric('min', 'min')
    assert mm.update(10)


def test_history(mm):
    assert mm.get_history() == [10.0, 11.0]


def test_to_dict(mm):
    start_time = mm.start_time
    conf = mm.to_config()
    assert conf['name'] == 'name'
    assert conf['best_value'] == 10
    assert conf['last_value'] == 11
    assert conf['history'] == [10, 11]
    assert conf['start_time'] == start_time


def test_to_dict_to_json_to_dict(mm):
    conf = mm.to_config()
    conf_json = json.loads(json.dumps(conf))
    assert conf_json == conf


def test_from_config_to_config(mm):
    mm.is_objective = True
    config = mm.to_config()
    mm2 = Metric.from_config(config)
    assert mm2.name == 'name'
    assert mm2.direction == 'min'
    assert mm2.get_history() == [10, 11]
    assert mm2.is_objective


def _test_classification_metrics(out):
    metrics = out["classification_metrics"]
    assert out["one_example_latency_millis"]
    assert np.isclose(.8, metrics["macro avg"]["f1-score"])
    assert np.isclose(.8, metrics["weighted avg"]["f1-score"])


class FakeModel(object):
    """A Fake 'Model' which simply returns the inputs provided after a minimal
    delay, used to simplify testing the metrics functionality."""

    def predict(self, x, batch_size=1):
        # Minimal sleep to ensure a non-zero inference latency calculation.
        time.sleep(.001)
        return x


@pytest.fixture
def fake_model():
    return FakeModel()


@pytest.fixture
def regression_data():
    """Generates a dataset for a regression-style problem with a single float
    output ranging from 0.0 to 1.0."""
    x_val = np.array([[0.0], [0.0], [0.0], [0.0], [0.0], [1.0], [1.0], [1.0],
                      [1.0], [1.0]])
    y_val = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0])
    return x_val, y_val


@pytest.fixture
def binary_classification_data():
    """Generates a binary classification dataset, with two float outputs
    representing the probability of each class."""

    x_val = np.array([[0, 1] for _ in range(5)] + [[1, 0] for _ in range(5)],
                     dtype=np.float32)
    y_val = [0, 0, 0, 0, 1, 0, 1, 1, 1, 1]
    y_val = np.array([(x, 1 - x) for x in y_val], dtype=np.float32)
    return x_val, y_val


@pytest.fixture
def multiclass_n_class_classification_data():
    """Generates a multi-class classification dataset, with multiple float
    outputs representing the probability of each class."""

    x_val = [x % 5 for x in range(25)]
    y_val = x_val[:20] + [3, 4, 0, 1, 2]
    x_val = tf.keras.utils.to_categorical(x_val, num_classes=5)
    y_val = tf.keras.utils.to_categorical(y_val, num_classes=5)
    return x_val, y_val


def test_regression_metrics(fake_model, regression_data):
    x_val, y_val = regression_data

    results = compute_common_classification_metrics(fake_model, (x_val, y_val))

    _test_classification_metrics(results)


def test_continuous_single_classification_ROC(fake_model, regression_data):
    x_val, y_val = regression_data

    results = compute_common_classification_metrics(fake_model, (x_val, y_val))

    _test_classification_metrics(results)

    assert np.allclose(results["roc_curve"]["fpr"], [0, 0.2, 1], atol=.05)
    assert np.allclose(results["roc_curve"]["tpr"], [0, 0.8, 1], atol=.05)
    assert np.allclose(results["roc_curve"]["thresholds"], [2, 1, 0], atol=.05)


def test_continuous_single_classification_metrics_int(fake_model,
                                                      regression_data):
    x_val, y_val = regression_data
    x_val = x_val.astype(np.int32)
    y_val = y_val.astype(np.int32)

    results = compute_common_classification_metrics(fake_model, (x_val, y_val))

    _test_classification_metrics(results)


def test_continuous_multi_classification_metrics_int(
        fake_model, binary_classification_data):
    x_val, y_val = binary_classification_data
    x_val = x_val.astype(np.int32)

    results = compute_common_classification_metrics(fake_model, (x_val, y_val))

    _test_classification_metrics(results)


def test_continuous_multi_classification_metrics_float(
        fake_model, multiclass_n_class_classification_data):
    x_val, y_val = multiclass_n_class_classification_data

    results = compute_common_classification_metrics(fake_model, (x_val, y_val))

    _test_classification_metrics(results)


def test_continuous_multi_classification_metrics_5way(
        fake_model, multiclass_n_class_classification_data):
    x_val, y_val = multiclass_n_class_classification_data

    results = compute_common_classification_metrics(fake_model, (x_val, y_val))

    metrics = results["classification_metrics"]
    assert np.isclose(.8, metrics["macro avg"]["f1-score"])
    assert np.isclose(.8, metrics["weighted avg"]["f1-score"])
