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

"Variation of HyperBand algorithm."

from ...engine import tuner as tuner_module
from ...engine import oracle as oracle_module


class UltraBandOracle(oracle_module.Oracle):
    # TODO
    pass


class UltraBand(tuner_module.Tuner):

    def __init__(self,
                 hypermodel,
                 objective,
                 max_trials,
                 **kwargs):
        oracle = UltraBandOracle()
        super(UltraBand, self).__init__(
            oracle,
            hypermodel,
            objective,
            max_trials,
            **kwargs)
