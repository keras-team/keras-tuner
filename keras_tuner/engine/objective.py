# Copyright 2022 The KerasTuner Authors
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


class Objective:
    """The objective for optimization during tuning.

    Args:
        name: String. The name of the objective.
        direction: String. The value should be "min" or "max" indicating
            whether the objective value should be minimized or maximized.
    """

    def __init__(self, name, direction):
        self.name = name
        self.direction = direction

    def has_value(self, logs):
        """Check if objective value exists in logs.

        Args:
            logs: A dictionary with the metric names as the keys and the metric
                values as the values, which is in the same format as the `logs`
                argument for `Callback.on_epoch_end()`.

        Returns:
            Boolean, whether we can compute objective value from the logs.
        """
        return self.name in logs

    def get_value(self, logs):
        """Get the objective value from the metrics logs.

        Args:
            logs: A dictionary with the metric names as the keys and the metric
                values as the values, which is in the same format as the `logs`
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

    def __eq__(self, obj):
        return self.name == obj.name and self.direction == obj.direction


class DefaultObjective(Objective):
    """Default objective to minimize if not provided by the user."""

    def __init__(self):
        super().__init__(name="default_objective", direction="min")


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

    def has_value(self, logs):
        return all([key in logs for key in self.name_to_direction])

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

    def __eq__(self, obj):
        if self.name_to_direction.keys() != obj.name_to_direction.keys():
            return False
        return sorted(self.objectives, key=lambda x: x.name) == sorted(
            obj.objectives, key=lambda x: x.name
        )


def create_objective(objective):
    if objective is None:
        return DefaultObjective()
    if isinstance(objective, list):
        return MultiObjective([create_objective(obj) for obj in objective])
    if isinstance(objective, Objective):
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
        return Objective(name=objective, direction=direction)
    else:
        raise ValueError(
            "`objective` not understood, expected str or "
            "`Objective` object, found: {}".format(objective)
        )
