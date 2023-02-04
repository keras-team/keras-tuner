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


from keras_tuner import applications
from keras_tuner import oracles
from keras_tuner import tuners
from keras_tuner.engine.hypermodel import HyperModel
from keras_tuner.engine.hyperparameters import HyperParameter
from keras_tuner.engine.hyperparameters import HyperParameters
from keras_tuner.engine.logger import CloudLogger
from keras_tuner.engine.logger import Logger
from keras_tuner.engine.objective import Objective
from keras_tuner.engine.oracle import Oracle
from keras_tuner.engine.oracle import synchronized
from keras_tuner.engine.tuner import Tuner
from keras_tuner.tuners import BayesianOptimization
from keras_tuner.tuners import GridSearch
from keras_tuner.tuners import Hyperband
from keras_tuner.tuners import RandomSearch
from keras_tuner.tuners import SklearnTuner
from keras_tuner.utils import check_tf_version
from keras_tuner.api_export import keras_tuner_export

check_tf_version()


class Version(str):
    __name__ = "Version"


__version__ = Version("1.2.1dev")
keras_tuner_export("keras_tuner.__version__")(__version__)
