
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
"""Tests for HyperResnet Model."""

import pytest

from kerastuner.applications import resnet
from kerastuner.engine import hyperparameters as hp_module


def test_model_construction():
    hp = hp_module.HyperParameters()
    hypermodel = resnet.HyperResnet(input_shape=(256, 256, 3), num_classes=10)
    model = hypermodel.build(hp)
    assert model.layers
    assert model.name == 'Resnet'
    assert model.output_shape == (None, 10)


def test_hyperparameter_existence_and_defaults():
    hp = hp_module.HyperParameters()
    hypermodel = resnet.HyperResnet(input_shape=(256, 256, 3), num_classes=10)
    model = hypermodel.build(hp)
    assert hp.get('version') == 'v2'
    assert hp.get('v2/conv3_depth') == 4
    assert hp.get('v2/conv4_depth') == 6
    assert hp.get('learning_rate') == 0.01
    # Pooling only set if `include_top=False`.
    assert 'pooling' not in hp.values


def test_include_top_false():
    hp = hp_module.HyperParameters()
    hypermodel = resnet.HyperResnet(input_shape=(256, 256, 3), num_classes=10, include_top=False)
    model = hypermodel.build(hp)
    assert hp.get('pooling') == 'avg'


def test_hyperparameter_override():
    hp = hp_module.HyperParameters()
    hp.Choice('version', ['v1'])
    hypermodel = resnet.HyperResnet(input_shape=(256, 256, 3), num_classes=10)
    model = hypermodel.build(hp)
    assert hp.get('version') == 'v1'
    assert hp.get('v1/conv3_depth') == 4
    assert hp.get('v1/conv4_depth') == 6
