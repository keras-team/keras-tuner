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

import pytest
from kerastuner.distributions.distributions import Distributions


def test_record_retrieve_param():
    dist = Distributions('test', {})
    name = 'param1'
    value = 3713
    group = 'group'
    key = dist._get_key(name, group)

    # record
    dist._record_hyperparameter(name, value, group)

    # retrieve
    hparams = dist.get_hyperparameters()
    assert key in hparams
    assert hparams[key]['name'] == name
    assert hparams[key]['value'] == value
    assert hparams[key]['group'] == group


def test_record_retrieve_config():
    hparam_config = {
        "this": 'that'
    }

    dist = Distributions('test', hparam_config)
    hparam_config2 = dist.get_hyperparameters_config()
    assert hparam_config == hparam_config2
