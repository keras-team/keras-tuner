# Copyright 2019 The KerasTuner Authors
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

from keras_tuner.engine import metrics_tracking
from keras_tuner.engine import objective as obj_module


class Objective(object):
    """The objective for optimization during tuning.

    Args:
        name: String. The name of the objective.
        direction: String. The value should be "min" or "max" indicating the
            objective value should be minimized or maximized.
    """

    def __init__(self, name, direction):
        self.name = name
        self.direction = direction

    def get_value(self, logs):
        """Get the objective value from the metrics logs.

        Args:
            logs: A dictionary with the metric names as the keys and the metric
                values as the values, which is the same format as the `logs`
                argument for `Callback.on_epoch_end()`.

        Returns:
            The objective value.
        """
        return logs[self.name]

    def better_than(self, a, b):
        """Whether the first objective value is better than the second.

        Args:
            a: A float, an objective value to compare.
            b: A float, another objective value to compare.

        Returns:
            Boolean, whether the first objective value is better than the
            second.
        """
        return (a > b and self.direction == "max") or (
            a < b and self.direction == "min"
        )


class MultiObjective(Objective):
    """A container for a list of objectives.

    Args:
        objectives: A list of `Objective`s.
    """

    def __init__(self, objectives):
        super().__init__(name="multi_objective", direction="min")
        self.objectives = objectives
        self.name_to_direction = {
            objective.name: objective.direction for objective in self.objectives
        }

    def get_value(self, logs):
        obj_value = 0
        for metric_name, metric_value in logs.items():
            if metric_name not in self.name_to_direction:
                continue
            if self.name_to_direction[metric_name] == "min":
                obj_value += metric_value
            else:
                obj_value -= metric_value
        return obj_value


def create_objective(objective):
    if objective is None:
        return obj_module.Objective("default_objective", "min")
    if isinstance(objective, list):
        return MultiObjective([create_objective(obj) for obj in objective])
    if isinstance(objective, obj_module.Objective):
        return objective
    if isinstance(objective, str):
        direction = metrics_tracking.infer_metric_direction(objective)
        if direction is None:
            error_msg = (
                'Could not infer optimization direction ("min" or "max") '
                'for unknown metric "{obj}". Please specify the objective  as'
                "a `keras_tuner.Objective`, for example `keras_tuner.Objective("
                '"{obj}", direction="min")`.'
            )
            error_msg = error_msg.format(obj=objective)
            raise ValueError(error_msg)
        return obj_module.Objective(name=objective, direction=direction)
    else:
        raise ValueError(
            "`objective` not understood, expected str or "
            "`Objective` object, found: {}".format(objective)
        )
