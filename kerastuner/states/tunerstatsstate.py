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

from __future__ import absolute_import
from time import time

from .state import State
from kerastuner.abstractions.display import fatal, subsection, display_settings


class TunerStatsState(State):
    "Track hypertuner statistics"

    def __init__(self):
        super(TunerStatsState, self).__init__()
        self.generated_instances = 0  # overall number of instances generated
        self.invalid_instances = 0  # how many models didn't work
        self.instances_previously_trained = 0  # num instance already trained
        self.collisions = 0  # how many time we regenerated the same model
        self.over_sized_models = 0  # num models with params> max_params

    def summary(self, extended=False):
        "display statistics summary"
        subsection("Tuning stats")
        display_settings(self.get_config())

    def get_config(self):
        return {
            'num_generated_models': self.generated_instances,
            'num_invalid_models': self.invalid_instances,
            "num_mdl_previously_trained": self.instances_previously_trained,
            "num_collision": self.collisions,
            "num_over_sized_models": self.over_sized_models
        }
