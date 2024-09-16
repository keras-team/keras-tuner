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

import inspect
from typing import Dict
from typing import List
from typing import Union

import numpy as np
import six

from keras_tuner import protos
from keras_tuner.api_export import keras_tuner_export
from keras_tuner.backend import keras


class ExecutionMetric:
    """Metric value at a given execution.

    If the model is trained multiple
    times (multiple executions), KerasTuner records the value of each
    metric at each training step. These values are aggregated
    over multiple executions into a list where each value corresponds
    to one execution.

    Args:
        value: The evaluated metric values.
        step: Int. The step of the evaluation, for example, the epoch number.
    """

    def __init__(self, value: Union[float | list[float]]):
        if not isinstance(value, list):
            value = [value]
        self.value = value

    def append(self, value: Union[float | list[float]]):
        if not isinstance(value, list):
            value = [value]
        self.value += value

    def get_config(self):
        return self.value

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def __eq__(self, other):
        return (
            other.value == self.value
            if isinstance(other, ExecutionMetric)
            else False
        )

    def __repr__(self):
        return f"ExecutionMetric(value={self.value})"

    def to_proto(self):
        return protos.get_proto().ExecutionMetric(value=self.value)

    @classmethod
    def from_proto(cls, proto):
        return cls(value=list(proto.value))


class MetricHistory:
    """Record of multiple executions of a single metric.

    It contains a collection of `ExecutionMetric` instances.

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
        self._current_best_value: Union[float, None] = None
        self._executions: List[ExecutionMetric] = []

    def append_execution(self, value):
        self._executions.append(ExecutionMetric(value))

    def get_best_value(self):
        # store best value in state, to make comparison fast.
        last_values = self.get_last_value()
        if not last_values and not self._current_best_value:
            return None

        # we update this value if necessary.
        current = self._current_best_value
        if self.direction == "min":
            last = float(np.nanmin(last_values))
            if current is None or last < current:
                current = last
        else:
            last = float(np.nanmax(last_values))
            if current is None or last > current:
                current = last

        self._current_best_value = current
        return current

    def get_best_location(self):
        # returns a 2D location.
        best_value = self.get_best_value()
        if best_value is None:
            return None

        for exec_idx, values in enumerate(self.get_executions_values()):
            for val_idx, value in enumerate(values):
                if value == best_value:
                    return (exec_idx, val_idx)

    def get_executions_values(self):
        if len(self._executions) > 0:
            return [execution.value for execution in self._executions]
        return None

    def get_history(self):
        return self._executions

    def get_statistics(self):
        values = self.get_executions_values()
        if len(values) != 0:
            return {
                "min": float(np.nanmin(values, (0, 1))),
                "max": float(np.nanmax(values, (0, 1))),
                "mean": float(np.nanmean(values, (1))),
                "median": float(np.nanmedian(values, (1))),
                "var": float(np.nanvar(values, (1))),
                "std": float(np.nanstd(values, (1))),
            }
        else:
            return None

    def get_last_value(self):
        values = self.get_executions_values()
        if isinstance(values, list) and len(values) != 0:
            return values[-1]
        else:
            return None

    def get_config(self):
        config = {
            "direction": self.direction,
            "executions": [obs.get_config() for obs in self.get_history()],
        }

        return config

    @classmethod
    def from_config(cls, config):
        instance = cls(config["direction"])
        instance.set_history(
            [ExecutionMetric.from_config(obs) for obs in config["executions"]]
        )
        return instance

    def to_proto(self):
        return protos.get_proto().MetricHistory(
            executions=[obs.to_proto() for obs in self.get_executions_values()],
            maximize=self.direction == "max",
        )

    @classmethod
    def from_proto(cls, proto):
        direction = "max" if proto.maximize else "min"
        instance = cls(direction)
        instance.set_history(
            [ExecutionMetric.from_proto(p) for p in proto.executions]
        )
        return instance


class MetricsTracker:
    """Record of the values of multiple executions of all metrics.

    It contains `MetricHistory` instances for the metrics. An "all-tracker".

    Args:
        metrics: List of strings of the names of the metrics.
    """

    def __init__(self, metrics=None):
        # str -> MetricHistory
        self.metrics: Dict[str, MetricHistory] = {}
        self.register_metrics(metrics)

    def exists(self, name: str):
        return name in self.metrics

    def register_metrics(self, metrics=None):
        metrics = metrics or []
        for metric in metrics:
            self.register(metric.name)

    def register(self, name: str, direction=None):
        if self.exists(name):
            raise ValueError(f"Metric already exists: {name}")
        if direction is None:
            direction = infer_metric_direction(name)
        if direction is None:
            # Objective direction is handled separately, but
            # non-objective direction defaults to min.
            direction = "min"
        self.metrics[name] = MetricHistory(direction)

    def update(self, name: str, value: Union[float, list[float]]):
        value = (
            [float(v) for v in value]
            if isinstance(value, list)
            else [float(value)]
        )
        if not self.exists(name):
            self.register(name)

        prev_best = self.metrics[name]._current_best_value
        self.metrics[name].append_execution(value)
        new_best = self.metrics[name].get_best_value()

        improved = new_best != prev_best
        return improved

    def get_history(self, name):
        self._assert_exists(name)
        return self.metrics[name].get_executions_values()

    def set_history(self, name: str, execution: Union[List[float], float]):
        if not self.exists(name):
            self.register(name)
        self.metrics[name].append_execution(execution)

    def get_best_value(self, name: str):
        self._assert_exists(name)
        return self.metrics[name].get_best_value()

    def get_best_step(self, name: str):
        self._assert_exists(name)
        return self.metrics[name].get_best_location()

    def get_statistics(self, name: str):
        self._assert_exists(name)
        return self.metrics[name].get_statistics()

    def get_last_value(self, name: str):
        self._assert_exists(name)
        return self.metrics[name].get_last_value()

    def get_direction(self, name: str):
        self._assert_exists(name)
        return self.metrics[name].direction

    def get_config(self):
        return {
            name: metric_history.get_config()
            for name, metric_history in self.metrics.items()
        }

    @classmethod
    def from_config(cls, config):
        instance = cls()
        instance.metrics = {
            name: MetricHistory.from_config(metric_history)
            for name, metric_history in config.items()
        }
        return instance

    def to_proto(self):
        return protos.get_proto().MetricsTracker(
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


@keras_tuner_export(
    "keras_tuner.engine.metrics_tracking.infer_metric_direction",
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
            if (
                "use_legacy_format"
                in inspect.getfullargspec(keras.metrics.deserialize).args
            ):
                metric = keras.metrics.deserialize(  # pragma: no cover
                    metric_name, use_legacy_format=True
                )
            else:
                metric = keras.metrics.deserialize(  # pragma: no cover
                    metric_name
                )
        except ValueError:
            try:
                if (
                    "use_legacy_format"
                    in inspect.getfullargspec(keras.losses.deserialize).args
                ):
                    metric = keras.losses.deserialize(  # pragma: no cover
                        metric_name, use_legacy_format=True
                    )
                else:
                    metric = keras.losses.deserialize(  # pragma: no cover
                        metric_name
                    )
            except Exception:
                # Direction can't be inferred.
                return None

    # Metric class, Loss class, or function.
    if isinstance(metric, (keras.metrics.Metric, keras.losses.Loss)):
        name = metric.__class__.__name__
        if name == "MeanMetricWrapper":
            name = metric._fn.__name__  # pragma: no cover
    elif isinstance(metric, str):
        name = metric
    else:
        name = metric.__name__

    if name in _MAX_METRICS or name in _MAX_METRIC_FNS:
        return "max"
    elif hasattr(keras.metrics, name) or hasattr(keras.losses, name):
        return "min"

    # Direction can't be inferred.
    return None
