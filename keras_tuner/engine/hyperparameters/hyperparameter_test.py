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

from keras_tuner.engine import hyperparameters as hp_module


def test_base_hyperparameter():
    base_param = hp_module.HyperParameter(name="base", default=0)
    assert base_param.name == "base"
    assert base_param.default == 0
    assert base_param.get_config() == {
        "name": "base",
        "default": 0,
        "conditions": [],
    }
    base_param = hp_module.HyperParameter.from_config(base_param.get_config())
    assert base_param.name == "base"
    assert base_param.default == 0
