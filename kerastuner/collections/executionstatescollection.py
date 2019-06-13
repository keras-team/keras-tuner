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

from .collections import Collection
from kerastuner.states.executionstate import ExecutionState
from kerastuner.abstractions.display import warning


class ExecutionStatesCollection(Collection):
    "Manage a collection of ExecutionStates"

    def __init__(self):
        super(ExecutionStatesCollection, self).__init__()

    def sort_by_metric(self, metric_name):
        "Returns a list of `ExecutionState`s sorted by a given metric."
        # FIXME: Refactor to avoid dup with InstanceState by adding an
        # Statescollection

        # Checking if the metric exists and get its direction.
        execution_state = self.get_last()
        # !don't use _objects -> use get() instead due to canonicalization
        metric = execution_state.metrics.get(metric_name)
        if not metric:
            warning('Metric %s not found' % metric_name)
            return []

        # getting metric values
        values = {}
        for execution_state in self._objects.values():
            value = execution_state.metrics.get(metric.name).get_best_value()
            # seems wrong but make it easy to sort by value and return excution
            values[value] = execution_state

        # sorting
        if metric.direction == 'min':
            sorted_values = sorted(values.keys())
        else:
            sorted_values = sorted(values.keys(), reverse=True)

        sorted_execution_states = []
        for val in sorted_values:
            sorted_execution_states.append(values[val])

        return sorted_execution_states

    @staticmethod
    def from_config(config):
        obj = ExecutionStatesCollection()
        for execution_state in config["execution_states"]:
            execution_state = ExecutionState.from_config(execution_state)
            obj.add(execution_state.idx, execution_state)
        obj._last_insert_idx = config["last_insert_idx"]
        return obj

    def get_config(self):
        config = {
            "execution_states": [],
            "last_insert_idx": self._last_insert_idx
        }
        for obj in self._objects.values():
            config["execution_states"].append(obj.get_config())

        return config
