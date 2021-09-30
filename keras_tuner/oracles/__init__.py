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

# Keep the name of `BayesianOptimization`, `Hyperband` and `RandomSearch`
# for backward compatibility for 1.0.2 or earlier.
from keras_tuner.tuners.bayesian import BayesianOptimizationOracle
from keras_tuner.tuners.hyperband import HyperbandOracle
from keras_tuner.tuners.hyperband import HyperbandOracle as Hyperband
from keras_tuner.tuners.randomsearch import RandomSearchOracle
from keras_tuner.tuners.randomsearch import RandomSearchOracle as RandomSearch

from keras_tuner.tuners.bayesian import (  # isort:skip
    BayesianOptimizationOracle as BayesianOptimization,
)
