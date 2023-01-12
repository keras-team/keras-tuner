# Copyright 2019 The KerasTuner Authors
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


import numpy as np
import six
from tensorflow import keras

from keras_tuner.protos import keras_tuner_pb2


class MetricObservation:
    """Metric value at a given step of training across multiple executions.

    If the model is trained multiple
    times (multiple executions), KerasTuner records the value of each
    metric at each training step. These values are aggregated
    over multiple executions into a list where each value corresponds
    to one execution.

    Args:
        value: Float or a list of floats. The evaluated metric values.
        step: Int. The step of the evaluation, for example, the epoch number.
    """

    def __init__(self, value, step):
        if not isinstance(value, list):
            value = [value]
        self.value = value
        self.step = step

    def append(self, value):
        if not isinstance(value, list):
            value = [value]
        self.value += value

    def mean(self):
        return np.mean(self.value)

    def get_config(self):
        return {"value": self.value, "step": self.step}

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def __eq__(self, other):
        return (
            other.value == self.value and other.step == self.step
            if isinstance(other, MetricObservation)
            else False
        )

    def __repr__(self):
        return f"MetricObservation(value={self.value}, step={self.step})"

    def to_proto(self):
        return keras_tuner_pb2.MetricObservation(value=self.value, step=self.step)

    @classmethod
    def from_proto(cls, proto):
        return cls(value=list(proto.value), step=proto.step)


class MetricHistory:
    """Record of multiple executions of a single metric.

    It contains a collection of `MetricObservation` instances.

    Args:
        direction: String. The direction of the metric to optimize. The value
            should be "min" or "max".
    """

    def __init__(self, direction="min"):
        if direction not in {"min", "max"}:
            raise ValueError(
                "`direction` should be one of "
                '{"min", "max"}, but got: %s' % (direction,)
            )
        self.direction = direction
        # Mapping step to `MetricObservation`.
        self._observations = {}

    def update(self, value, step):
        if step in self._observations:
            self._observations[step].append(value)
        else:
            self._observations[step] = MetricObservation(value, step=step)

    def get_best_value(self):
        values = [obs.mean() for obs in self._observations.values()]
        if not values:
            return None
        return np.nanmin(values) if self.direction == "min" else np.nanmax(values)

    def get_best_step(self):
        best_value = self.get_best_value()
        if best_value is None:
            return None
        for obs in self._observations.values():
            if obs.mean() == best_value:
                return obs.step

    def get_history(self):
        return sorted(self._observations.values(), key=lambda obs: obs.step)

    def set_history(self, observations):
        for obs in observations:
            self.update(obs.value, step=obs.step)

    def get_statistics(self):
        history = self.get_history()
        history_values = [obs.mean() for obs in history]
        return (
            {
                "min": float(np.nanmin(history_values)),
                "max": float(np.nanmax(history_values)),
                "mean": float(np.nanmean(history_values)),
                "median": float(np.nanmedian(history_values)),
                "var": float(np.nanvar(history_values)),
                "std": float(np.nanstd(history_values)),
            }
            if len(history_values)
            else {}
        )

    def get_last_value(self):
        history = self.get_history()
        if history:
            last_obs = history[-1]
            return last_obs.mean()
        else:
            return None

    def get_config(self):
        config = {
            "direction": self.direction,
            "observations": [obs.get_config() for obs in self.get_history()],
        }

        return config

    @classmethod
    def from_config(cls, config):
        instance = cls(config["direction"])
        instance.set_history(
            [MetricObservation.from_config(obs) for obs in config["observations"]]
        )
        return instance

    def to_proto(self):
        return keras_tuner_pb2.MetricHistory(
            observations=[obs.to_proto() for obs in self.get_history()],
            maximize=self.direction == "max",
        )

    @classmethod
    def from_proto(cls, proto):
        direction = "max" if proto.maximize else "min"
        instance = cls(direction)
        instance.set_history(
            [MetricObservation.from_proto(p) for p in proto.observations]
        )
        return instance


class MetricsTracker:
    """Record of the values of multiple executions of all metrics.

    It contains `MetricHistory` instances for the metrics.

    Args:
        metrics: List of strings of the names of the metrics.
    """

    def __init__(self, metrics=None):
        # str -> MetricHistory
        self.metrics = {}
        self.register_metrics(metrics)

    def exists(self, name):
        return name in self.metrics

    def register_metrics(self, metrics=None):
        metrics = metrics or []
        for metric in metrics:
            self.register(metric.name)

    def register(self, name, direction=None):
        if self.exists(name):
            raise ValueError(f"Metric already exists: {name}")
        if direction is None:
            direction = infer_metric_direction(name)
        if direction is None:
            # Objective direction is handled separately, but
            # non-objective direction defaults to min.
            direction = "min"
        self.metrics[name] = MetricHistory(direction)

    def update(self, name, value, step=0):
        value = float(value)
        if not self.exists(name):
            self.register(name)

        prev_best = self.metrics[name].get_best_value()
        self.metrics[name].update(value, step=step)
        new_best = self.metrics[name].get_best_value()

        improved = new_best != prev_best
        return improved

    def get_history(self, name):
        self._assert_exists(name)
        return self.metrics[name].get_history()

    def set_history(self, name, observations):
        assert type(observations) == list
        if not self.exists(name):
            self.register(name)
        self.metrics[name].set_history(observations)

    def get_best_value(self, name):
        self._assert_exists(name)
        return self.metrics[name].get_best_value()

    def get_best_step(self, name):
        self._assert_exists(name)
        return self.metrics[name].get_best_step()

    def get_statistics(self, name):
        self._assert_exists(name)
        return self.metrics[name].get_statistics()

    def get_last_value(self, name):
        self._assert_exists(name)
        return self.metrics[name].get_last_value()

    def get_direction(self, name):
        self._assert_exists(name)
        return self.metrics[name].direction

    def get_config(self):
        return {
            "metrics": {
                name: metric_history.get_config()
                for name, metric_history in self.metrics.items()
            }
        }

    @classmethod
    def from_config(cls, config):
        instance = cls()
        instance.metrics = {
            name: MetricHistory.from_config(metric_history)
            for name, metric_history in config["metrics"].items()
        }
        return instance

    def to_proto(self):
        return keras_tuner_pb2.MetricsTracker(
            metrics={
                name: metric_history.to_proto()
                for name, metric_history in self.metrics.items()
            }
        )

    @classmethod
    def from_proto(cls, proto):
        instance = cls()
        instance.metrics = {
            name: MetricHistory.from_proto(metric_history)
            for name, metric_history in proto.metrics.items()
        }
        return instance

    def _assert_exists(self, name):
        if name not in self.metrics:
            raise ValueError(f"Unknown metric: {name}")


_MAX_METRICS = (
    "Accuracy",
    "BinaryAccuracy",
    "CategoricalAccuracy",
    "SparseCategoricalAccuracy",
    "TopKCategoricalAccuracy",
    "SparseTopKCategoricalAccuracy",
    "TruePositives",
    "TrueNegatives",
    "Precision",
    "Recall",
    "AUC",
    "SensitivityAtSpecificity",
    "SpecificityAtSensitivity",
)

_MAX_METRIC_FNS = (
    "accuracy",
    "categorical_accuracy",
    "binary_accuracy",
    "sparse_categorical_accuracy",
)


def infer_metric_direction(metric):
    # Handle str input and get canonical object.
    if isinstance(metric, six.string_types):
        metric_name = metric

        if metric_name.startswith("val_"):
            metric_name = metric_name.replace("val_", "", 1)

        if metric_name.startswith("weighted_"):
            metric_name = metric_name.replace("weighted_", "", 1)

        # Special-cases (from `keras/engine/training_utils.py`)
        if metric_name in {"loss", "crossentropy", "ce"}:
            return "min"
        elif metric_name == "acc":
            return "max"

        try:
            metric = keras.metrics.get(metric_name)
        except ValueError:
            try:
                metric = keras.losses.get(metric_name)
            except Exception:
                # Direction can't be inferred.
                return None

    # Metric class, Loss class, or function.
    if isinstance(metric, (keras.metrics.Metric, keras.losses.Loss)):
        name = metric.__class__.__name__
        if name == "MeanMetricWrapper":
            name = metric._fn.__name__
    else:
        name = metric.__name__

    if name in _MAX_METRICS or name in _MAX_METRIC_FNS:
        return "max"
    elif hasattr(keras.metrics, name) or hasattr(keras.losses, name):
        return "min"

    # Direction can't be inferred.
    return None
