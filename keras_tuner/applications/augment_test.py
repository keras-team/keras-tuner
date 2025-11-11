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
"""Tests for HyperImageAugment Models."""

import numpy as np
import pytest

from keras_tuner.applications import augment as aug_module
from keras_tuner.backend import config
from keras_tuner.backend import keras
from keras_tuner.engine import hyperparameters as hp_module


def test_transforms_search_space():
    hm = aug_module.HyperImageAugment(input_shape=(32, 32, 3))
    # Default choice
    assert hm.transforms == [
        ("rotate", (0, 0.5)),
        ("translate_x", (0, 0.4)),
        ("translate_y", (0, 0.4)),
        ("contrast", (0, 0.3)),
    ]

    hm = aug_module.HyperImageAugment(
        input_shape=(32, 32, 3),
        rotate=0.3,
        translate_x=[0.1, 0.5],
        contrast=None,
    )
    assert hm.transforms == [
        ("rotate", (0, 0.3)),
        ("translate_x", (0.1, 0.5)),
        ("translate_y", (0, 0.4)),
    ]


def test_input_requirement():
    hp = hp_module.HyperParameters()
    with pytest.raises(ValueError, match=r".*must specify.*"):
        hm = aug_module.HyperImageAugment()

    hm = aug_module.HyperImageAugment(input_shape=(None, None, 3))
    model = hm.build(hp)
    assert model.built

    hm = aug_module.HyperImageAugment(
        input_tensor=keras.Input(shape=(32, 32, 3))
    )
    model = hm.build(hp)
    assert model.built


def test_model_construction_factor_one():
    hp = hp_module.HyperParameters()
    hm = aug_module.HyperImageAugment(input_shape=(None, None, 3))
    model = hm.build(hp)
    # augment_layers search default space [0, 4], with default zero.
    assert len(model.layers) == 2

    hp = hp_module.HyperParameters()
    hm = aug_module.HyperImageAugment(
        input_shape=(None, None, 3), augment_layers=1
    )
    model = hm.build(hp)
    # factors default all zero, the model should only have input layer
    assert len(model.layers) == 2


def test_model_construction_fixed_aug():
    hp = hp_module.HyperParameters()
    hm = aug_module.HyperImageAugment(
        input_shape=(None, None, 3), rotate=[0.2, 0.5], augment_layers=0
    )
    model = hm.build(hp)
    assert model.layers
    assert model.name == "image_augment"

    # Output shape includes batch dimension.
    assert model.output_shape == (None, None, None, 3)
    out = model.predict(np.ones((1, 32, 32, 3)))
    assert out.shape == (1, 32, 32, 3)
    # Augment does not distort image when inferencing.
    assert (out != 1).sum() == 0


def test_hyperparameter_selection_and_hp_defaults_fixed_aug():
    hp = hp_module.HyperParameters()
    hm = aug_module.HyperImageAugment(
        input_shape=(32, 32, 3),
        translate_x=[0.2, 0.4],
        contrast=None,
        augment_layers=0,
    )
    hm.build(hp)
    # default value of default search space are always minimum.
    assert hp.get("factor_rotate") == 0
    assert hp.get("factor_translate_x") == 0.2
    assert hp.get("factor_translate_y") == 0
    assert "factor_contrast" not in hp.values


@pytest.mark.skipif(
    not config.multi_backend(),
    reason="The test fails with tf.keras.",
)
def test_hyperparameter_existence_and_hp_defaults_rand_aug():
    hp = hp_module.HyperParameters()
    hm = aug_module.HyperImageAugment(
        input_shape=(32, 32, 3), augment_layers=[2, 5], contrast=False
    )
    hm.build(hp)
    assert hp.get("augment_layers") == 2


def test_augment_layers_not_int():
    with pytest.raises(ValueError, match="must be int"):
        hp = hp_module.HyperParameters()
        hm = aug_module.HyperImageAugment(
            input_shape=(32, 32, 3), augment_layers=1.5
        )
        hm.build(hp)


@pytest.mark.skipif(
    not config.multi_backend(),
    reason="The test fails with tf.keras.",
)
def test_contrast_0_5_to_1():
    hp = hp_module.HyperParameters()
    hm = aug_module.HyperImageAugment(
        input_shape=(32, 32, 3), augment_layers=[1, 2], contrast=[0.5, 1]
    )
    hm.build(hp)


def test_3_range_params_raise_error():
    with pytest.raises(ValueError, match="exceed 2"):
        hp = hp_module.HyperParameters()
        hm = aug_module.HyperImageAugment(
            input_shape=(32, 32, 3), augment_layers=[1, 2], contrast=[0.5, 1, 2]
        )
        hm.build(hp)


def test_contrast_range_receive_string_raise_error():
    with pytest.raises(ValueError, match="must be int or float"):
        hp = hp_module.HyperParameters()
        hm = aug_module.HyperImageAugment(
            input_shape=(32, 32, 3), augment_layers=[1, 2], contrast=[0.5, "a"]
        )
        hm.build(hp)


def test_hyperparameter_override_fixed_aug():
    hp = hp_module.HyperParameters()
    hp.Fixed("factor_rotate", 0.9)
    hp.Choice("factor_translate_x", [0.8])
    hm = aug_module.HyperImageAugment(input_shape=(32, 32, 3), augment_layers=0)
    hm.build(hp)
    assert hp.get("factor_rotate") == 0.9
    assert hp.get("factor_translate_x") == 0.8
    assert hp.get("factor_translate_y") == 0.0
    assert hp.get("factor_contrast") == 0.0


@pytest.mark.skipif(
    not config.multi_backend(),
    reason="The test fails with tf.keras.",
)
def test_hyperparameter_override_rand_aug():
    hp = hp_module.HyperParameters()
    hp.Fixed("randaug_mag", 1.0)
    hp.Choice("randaug_count", [4])
    hm = aug_module.HyperImageAugment(
        input_shape=(32, 32, 3), augment_layers=[2, 4]
    )
    hm.build(hp)
    assert hp.get("randaug_mag") == 1.0
    assert hp.get("randaug_count") == 4
