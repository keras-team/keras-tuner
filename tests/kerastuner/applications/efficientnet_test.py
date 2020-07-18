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
"""Tests for HyperEfficientNet Model."""

import numpy as np
import os
import pytest
import tensorflow as tf

from kerastuner.applications import efficientnet 
from kerastuner.engine import hyperparameters as hp_module


@pytest.mark.skipif('TRAVIS' in os.environ, reason='Causes CI to stall')
@pytest.mark.parametrize('version', ['B0', 'B1'])
def test_model_construction(version):
    hp = hp_module.HyperParameters()
    hp.Choice('version', [version])
    hypermodel = efficientnet.HyperEfficientNet(input_shape=(32, 32, 3), classes=10)
    model = hypermodel.build(hp)
    assert hp.values['version'] == version
    assert model.layers
    assert model.name == 'EfficientNet'
    assert model.output_shape == (None, 10)
    model.train_on_batch(np.ones((1, 128, 128, 3)), np.ones((1, 10)))
    out = model.predict(np.ones((1, 128, 128, 3)))
    assert out.shape == (1, 10)


def test_hyperparameter_existence_and_defaults():
    hp = hp_module.HyperParameters()
    hypermodel = efficientnet.HyperEfficientNet(input_shape=(224, 224, 3), classes=10)
    hypermodel.build(hp)
    assert hp.get('version') == 'B0'
    assert hp.get('top_dropout_rate') == 0.2
    assert hp.get('learning_rate') == 0.01
    assert hp.get('pooling') == 'avg'


def test_hyperparameter_override():
    hp = hp_module.HyperParameters()
    hp.Choice('version', ['B1'])
    hp.Fixed('top_dropout_rate', 0.5)
    hypermodel = efficientnet.HyperEfficientNet(input_shape=(256, 256, 3), classes=10)
    hypermodel.build(hp)
    assert hp.get('version') == 'B1'
    assert hp.get('top_dropout_rate') == 0.5

def test_input_tensor():
    hp = hp_module.HyperParameters()
    inputs = tf.keras.Input(shape=(256, 256, 3))
    hypermodel = efficientnet.HyperEfficientNet(input_tensor=inputs, classes=10)
    model = hypermodel.build(hp)
    assert model.inputs == [inputs]

