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
import time

import numpy as np
import pytest
import sklearn
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model

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


def _single_output_model(dtype):
    i = Input(shape=(1, ), dtype=dtype)
    o = i
    m = Model(i, o)
    m.compile(loss="mse", optimizer="adam")
    return m


def _multi_output_model(n_input, dtype):
    i = Input(shape=(n_input, ), dtype=dtype, name="Foo")
    m = Model(i, i)
    m.compile(loss="mse", optimizer="adam")
    return m


def test_continuous_single_classification_metrics():
    model = _single_output_model(dtype=tf.float32)
    x_val = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    y_val = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0])

    results = compute_common_classification_metrics(model, (x_val, y_val))

    _test_classification_metrics(results)


def test_continuous_single_classification_metrics_int():
    model = _single_output_model(dtype=tf.float32)
    x_val = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    y_val = np.array([0, 0, 0, 0, 1, 0, 1, 1, 1, 1])

    results = compute_common_classification_metrics(model, (x_val, y_val))

    _test_classification_metrics(results)


def test_continuous_multi_classification_metrics():
    model = _multi_output_model(2, dtype=tf.float32)

    x_val = np.array([[-1, 1] for _ in range(5)] + [[1, -1] for _ in range(5)])

    y_val = [0, 0, 0, 0, 1, 0, 1, 1, 1, 1]
    y_val = np.array([(x, 1 - x) for x in y_val], dtype=np.float32)

    results = compute_common_classification_metrics(model, (x_val, y_val))

    _test_classification_metrics(results)


def test_continuous_multi_classification_metrics_float():
    model = _multi_output_model(2, dtype=tf.float32)

    model.summary()

    x_val = np.array([[0, 1] for _ in range(5)] + [[0, -1] for _ in range(5)],
                     dtype=np.float32)
    y_val = [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0]
    y_val = np.array([[x, 1 - x] for x in y_val], dtype=np.float32)
    results = compute_common_classification_metrics(model, (x_val, y_val))

    _test_classification_metrics(results)


def test_continuous_multi_classification_metrics_5way():
    model = _multi_output_model(5, dtype=tf.float32)

    x = [x for x in range(5)] + [-x for x in range(5)]
    x = tf.keras.utils.to_categorical(x, num_classes=5)
    y = x

    results = compute_common_classification_metrics(model, (x, y))

    metrics = results["classification_metrics"]
    assert np.isclose(1, metrics["macro avg"]["f1-score"])
    assert np.isclose(1, metrics["weighted avg"]["f1-score"])


def test_continuous_single_classification_metrics_training_end():
    model = _single_output_model(dtype=tf.float32)
    x_val = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    y_val = np.array([0, 0, 0, 0, 1, 0, 1, 1, 1, 1])

    results = compute_common_classification_metrics(model, (x_val, y_val))

    _test_classification_metrics(results)

    assert np.allclose(results["roc_curve"]["fpr"], [0, 0.2, 1], atol=.05)
    assert np.allclose(results["roc_curve"]["tpr"], [0, 0.8, 1], atol=.05)
    assert np.allclose(results["roc_curve"]["thresholds"], [2, 1, 0], atol=.05)
