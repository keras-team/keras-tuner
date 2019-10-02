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
"""Tests for HyperXception Model."""

import numpy as np
import os
import pytest
import tensorflow as tf

from kerastuner.applications import xception
from kerastuner.engine import hyperparameters as hp_module


@pytest.mark.skipif('TRAVIS' in os.environ, reason='Causes CI to stall')
@pytest.mark.parametrize('pooling', ['flatten', 'avg', 'max'])
def test_model_construction(pooling):
    hp = hp_module.HyperParameters()
    hp.Choice('pooling', [pooling])
    hypermodel = xception.HyperXception(
        input_shape=(128, 128, 3), classes=10)
    model = hypermodel.build(hp)
    assert hp.values['pooling'] == pooling
    assert model.layers
    assert model.name == 'Xception'
    assert model.output_shape == (None, 10)
    model.train_on_batch(np.ones((1, 128, 128, 3)), np.ones((1, 10)))
    out = model.predict(np.ones((1, 128, 128, 3)))
    assert out.shape == (1, 10)


def test_hyperparameter_existence_and_defaults():
    hp = hp_module.HyperParameters()
    hypermodel = xception.HyperXception(
        input_shape=(256, 256, 3), classes=10)
    hypermodel.build(hp)
    assert hp.values == {
        'activation': 'relu',
        'conv2d_num_filters': 64,
        'kernel_size': 3,
        'initial_strides': 2,
        'sep_num_filters': 256,
        'num_residual_blocks': 4,
        'pooling': 'avg',
        'num_dense_layers': 1,
        'dropout_rate': 0.5,
        'dense_use_bn': True,
        'learning_rate': 1e-3
    }


def test_include_top_false():
    hp = hp_module.HyperParameters()
    hypermodel = xception.HyperXception(
        input_shape=(256, 256, 3), classes=10, include_top=False)
    model = hypermodel.build(hp)
    assert not model.optimizer


def test_hyperparameter_override():
    hp = hp_module.HyperParameters()
    hp.Choice('pooling', ['flatten'])
    hp.Choice('num_dense_layers', [2])
    hypermodel = xception.HyperXception(
        input_shape=(256, 256, 3), classes=10)
    hypermodel.build(hp)
    assert hp.get('pooling') == 'flatten'
    assert hp.get('num_dense_layers') == 2


def test_input_tensor():
    hp = hp_module.HyperParameters()
    inputs = tf.keras.Input((256, 256, 3))
    hypermodel = xception.HyperXception(
        input_tensor=inputs, include_top=False)
    model = hypermodel.build(hp)
    assert model.inputs == [inputs]
