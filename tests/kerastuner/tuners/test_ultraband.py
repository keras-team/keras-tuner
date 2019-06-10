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

import json
import time

import numpy as np
import pytest
import sklearn
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model
from collections import namedtuple
from sklearn.utils.multiclass import type_of_target
from kerastuner.tuners.ultraband import UltraBand
from kerastuner.engine.metric import compute_common_classification_metrics


def basic_model_fn():
    # *can't pass as fixture as the tuner expects a function
    i = Input(shape=(1, ), name="input")
    o = Dense(4, name="output")(i)
    model = Model(inputs=i, outputs=o)
    model.compile(optimizer="adam", loss='mse', metrics=['accuracy'])
    return model


BracketInfo = namedtuple("BracketInfo",
                         ["num_models", "num_epochs", "num_to_keep"])


class CapturingUltraBand(UltraBand):
    def __init__(self, *args, **kwargs):
        super(CapturingUltraBand, self).__init__(*args, **kwargs)
        self.bracket_infos = []

    def bracket(self, instance_collection, num_to_keep, num_epochs,
                total_num_epochs, x, y, **fit_kwargs):
        self.bracket_infos.append(
            BracketInfo(num_models=len(instance_collection),
                        num_epochs=num_epochs,
                        num_to_keep=num_to_keep))
        return super(CapturingUltraBand,
                     self).bracket(instance_collection, num_to_keep,
                                   num_epochs, total_num_epochs, x, y,
                                   **fit_kwargs)

    def get_bracket_infos(self):
        return self.bracket_infos


@pytest.fixture
def ultraband():
    return CapturingUltraBand(basic_model_fn,
                              'loss',
                              dry_run=True,
                              epoch_budget=324,
                              min_epochs=3,
                              max_epochs=27,
                              model_name="ultraband_test")


def test_fake_ultraband(ultraband):
    x = []
    y = []

    ultraband.search(x, y)

    expected = [
        # Initial group
        BracketInfo(num_models=27, num_epochs=3, num_to_keep=9),
        BracketInfo(num_models=9, num_epochs=6, num_to_keep=3),
        BracketInfo(num_models=3, num_epochs=18, num_to_keep=3),

        # Final, partial group
        BracketInfo(num_models=19, num_epochs=3, num_to_keep=6),
        BracketInfo(num_models=6, num_epochs=6, num_to_keep=2),
        BracketInfo(num_models=2, num_epochs=18, num_to_keep=2)
    ]

    bracket_info = ultraband.get_bracket_infos()
    print(bracket_info)

    for i in range(len(expected)):
        assert bracket_info[i] == expected[i]

    assert np.array_equal(bracket_info, expected)
