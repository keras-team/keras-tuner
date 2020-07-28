# Copyright 2020 The Keras Tuner Authors
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
"""Tests for HyperAugment Models."""

import numpy as np
import os
import pytest
from tensorflow import keras

from kerastuner.applications import augment as aug_module
from kerastuner.engine import hyperparameters as hp_module


@pytest.mark.skipif('TRAVIS' in os.environ, reason='Causes CI to stall')
@pytest.mark.parametrize('augment',
                         [aug_module.HyperAugment,
                          aug_module.HyperFixedAugment,
                          aug_module.HyperRandAugment])
def test_transforms_choice(augment):
    hm = augment(input_shape=(32, 32, 3))
    # Default choice
    assert hm.transforms == ['rotate',
                             'translate_x',
                             'translate_y',
                             'contrast']

    hm = augment(input_shape=(32, 32, 3),
                 rotate=True,
                 translate_x=False,
                 contrast=False)
    assert hm.transforms == ['rotate', 'translate_y']


def test_input_requirement_fixed_aug():
    hp = hp_module.HyperParameters()
    # Allow not specifying input shape/tensor.
    hm = aug_module.HyperFixedAugment()
    model = hm.build(hp)
    assert not model.built


def test_input_requirement_rand_aug():
    hp = hp_module.HyperParameters()
    with pytest.raises(ValueError, match=r'.*must specify.*'):
        hm = aug_module.HyperRandAugment()

    hm = aug_module.HyperRandAugment(input_shape=(None, None, 3))
    model = hm.build(hp)
    assert model.built

    hm = aug_module.HyperRandAugment(input_tensor=keras.Input(shape=(32, 32, 3)))
    model = hm.build(hp)
    assert model.built


def test_model_construction_fixed_aug():
    hp = hp_module.HyperParameters()
    hm = aug_module.HyperFixedAugment(input_shape=(None, None, 3))
    model = hm.build(hp)
    assert model.layers
    assert model.name == 'fixed_augment'

    # Output shape includes batch dimension.
    assert model.output_shape == (None, None, None, 3)
    out = model.predict(np.ones((1, 32, 32, 3)))
    assert out.shape == (1, 32, 32, 3)
    # Augment does not distort image when inferencing.
    assert (out != 1).sum() == 0


def test_model_construction_rand_aug():
    hp = hp_module.HyperParameters()
    hm = aug_module.HyperRandAugment(input_shape=(None, None, 3))
    model = hm.build(hp)
    assert model.layers
    assert model.name == 'rand_augment'

    # Output shape includes batch dimension.
    assert model.output_shape == (None, None, None, 3)
    out = model.predict(np.ones((1, 32, 32, 3)))
    assert out.shape == (1, 32, 32, 3)
    # Augment does not distort image when inferencing.
    assert (out != 1).sum() == 0


def test_hyperparameter_selection_and_defaults_fixed_aug():
    hp = hp_module.HyperParameters()
    hm = aug_module.HyperFixedAugment(input_shape=(32, 32, 3),
                                      contrast=False)
    hm.build(hp)
    assert hp.get('factor_rotate') == 0.15
    assert hp.get('factor_translate_x') == 0.15
    assert hp.get('factor_translate_y') == 0.15
    assert 'factor_contrast' not in hp.values


def test_hyperparameter_existence_and_defaults_rand_aug():
    hp = hp_module.HyperParameters()
    hm = aug_module.HyperRandAugment(input_shape=(32, 32, 3),
                                     contrast=False)
    hm.build(hp)
    assert hp.get('randaug_mag') == 0.2
    assert hp.get('randaug_count') == 2


def test_hyperparameter_override_fixed_aug():
    hp = hp_module.HyperParameters()
    hp.Fixed('factor_rotate', 0.9)
    hp.Choice('factor_translate_x', [0.8])
    hm = aug_module.HyperFixedAugment(input_shape=(32, 32, 3))
    hm.build(hp)
    assert hp.get('factor_rotate') == 0.9
    assert hp.get('factor_translate_x') == 0.8
    assert hp.get('factor_translate_y') == 0.15
    assert hp.get('factor_contrast') == 0.15


def test_hyperparameter_override_rand_aug():
    hp = hp_module.HyperParameters()
    hp.Fixed('randaug_mag', 1.0)
    hp.Choice('randaug_count', [4])
    hm = aug_module.HyperRandAugment(input_shape=(32, 32, 3))
    hm.build(hp)
    assert hp.get('randaug_mag') == 1.0
    assert hp.get('randaug_count') == 4
