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
"""Tests for HyperEfficientNet Model."""

import numpy as np
import pytest

from keras_tuner.applications import efficientnet
from keras_tuner.backend import config
from keras_tuner.backend import keras
from keras_tuner.engine import hypermodel as hm_module
from keras_tuner.engine import hyperparameters as hp_module

if config.backend() == "torch":
    keras.backend.set_image_data_format("channels_first")
else:
    keras.backend.set_image_data_format("channels_last")

if keras.backend.image_data_format() == "channels_last":
    INPUT_SHAPE_32 = (32, 32, 3)
    INPUT_SHAPE_224 = (224, 224, 3)
    INPUT_SHAPE_256 = (256, 256, 3)
else:
    INPUT_SHAPE_32 = (3, 32, 32)
    INPUT_SHAPE_224 = (3, 224, 224)
    INPUT_SHAPE_256 = (3, 256, 256)


@pytest.mark.parametrize("version", ["B0", "B1"])
@pytest.mark.skipif(
    config.multi_backend(),
    reason="The test is too slow.",
)
def test_model_construction(version):
    hp = hp_module.HyperParameters()
    hp.Choice("version", [version])
    hypermodel = efficientnet.HyperEfficientNet(
        input_shape=INPUT_SHAPE_32, classes=10
    )
    model = hypermodel.build(hp)
    assert hp.values["version"] == version
    assert model.layers
    assert model.name == "EfficientNet"
    assert model.output_shape == (None, 10)
    model.build(input_shape=(32, 32, 3))
    model.train_on_batch(np.ones((1,) + INPUT_SHAPE_32), np.ones((1, 10)))
    out = model.predict(np.ones((1,) + INPUT_SHAPE_32))
    assert out.shape == (1, 10)


def test_hyperparameter_existence_and_defaults():
    hp = hp_module.HyperParameters()
    hypermodel = efficientnet.HyperEfficientNet(
        input_shape=INPUT_SHAPE_224, classes=10
    )
    hypermodel.build(hp)
    assert hp.get("version") == "B0"
    assert hp.get("top_dropout_rate") == 0.2
    assert hp.get("learning_rate") == 0.01
    assert hp.get("pooling") == "avg"


def test_hyperparameter_override():
    hp = hp_module.HyperParameters()
    hp.Choice("version", ["B1"])
    hp.Fixed("top_dropout_rate", 0.5)
    hypermodel = efficientnet.HyperEfficientNet(
        input_shape=INPUT_SHAPE_256, classes=10
    )
    hypermodel.build(hp)
    assert hp.get("version") == "B1"
    assert hp.get("top_dropout_rate") == 0.5


def test_input_tensor():
    hp = hp_module.HyperParameters()
    inputs = keras.Input(shape=INPUT_SHAPE_256)
    hypermodel = efficientnet.HyperEfficientNet(input_tensor=inputs, classes=10)
    model = hypermodel.build(hp)
    assert model.inputs == [inputs]


def test_override_compiling_phase():
    class MyHyperEfficientNet(efficientnet.HyperEfficientNet):
        def _compile(self, model, hp):
            learning_rate = 0.1
            optimizer_name = hp.Choice(
                "optimizer", ["adam", "sgd"], default="adam"
            )
            if optimizer_name == "sgd":
                optimizer = keras.optimizers.SGD(
                    momentum=0.1, learning_rate=learning_rate
                )
            elif optimizer_name == "adam":
                optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
            model.compile(
                optimizer=optimizer,
                loss="categorical_crossentropy",
                metrics=["accuracy"],
            )

    hp = hp_module.HyperParameters()
    hypermodel = MyHyperEfficientNet(input_shape=INPUT_SHAPE_32, classes=5)
    hypermodel.build(hp)
    assert "learning_rate" not in hp.values
    assert hp.values["optimizer"] == "adam"


def test_augmentation_param_invalid_input():
    with pytest.raises(ValueError):
        efficientnet.HyperEfficientNet(
            input_shape=INPUT_SHAPE_32, classes=10, augmentation_model=0
        )


def test_augmentation_param_fixed_model():
    hp = hp_module.HyperParameters()
    aug_model = keras.Sequential([keras.layers.RandomRotation(1.0)], name="aug")
    hypermodel = efficientnet.HyperEfficientNet(
        input_shape=INPUT_SHAPE_32, classes=10, augmentation_model=aug_model
    )
    model = hypermodel.build(hp)
    assert model.layers[1].name == "aug"


def test_augmentation_param_hyper_model():
    class HyperAug(hm_module.HyperModel):
        def build(self, hp):
            model = keras.Sequential(name="aug")
            scaling_factor = hp.Choice("scaling_factor", [1])
            model.add(keras.layers.Lambda(lambda x: x * scaling_factor))
            return model

    hp = hp_module.HyperParameters()
    aug_hm = HyperAug()
    hypermodel = efficientnet.HyperEfficientNet(
        input_shape=INPUT_SHAPE_32, classes=10, augmentation_model=aug_hm
    )
    model = hypermodel.build(hp)
    assert model.layers[1].name == "aug"
    assert hp.values["scaling_factor"] == 1


def test_pooling_is_max():
    hp = hp_module.HyperParameters()
    hp.values["pooling"] = "max"
    hypermodel = efficientnet.HyperEfficientNet(
        input_shape=INPUT_SHAPE_32, classes=10
    )
    hypermodel.build(hp)


def test_no_classes_raise_error():
    with pytest.raises(ValueError, match="classes"):
        efficientnet.HyperEfficientNet(input_shape=INPUT_SHAPE_32)


def test_no_input_shape_tensor_raise_error():
    with pytest.raises(ValueError, match="input_tensor"):
        efficientnet.HyperEfficientNet(classes=10)
