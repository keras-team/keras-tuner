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
import numpy as np

import kerastuner
from kerastuner.engine import tuner as tuner_module

from tensorflow import keras

"""TODO:
test correctness of per-execution metric tracking
test correctness of per-trial metric tracking (averaging)
test correctness of tuner metric tracking (best values)
test correctness of get_best_model
"""

INPUT_DIM = 2
NUM_CLASSES = 3
NUM_SAMPLES = 64
TRAIN_INPUTS = np.random.random(size=(NUM_SAMPLES, INPUT_DIM))
TRAIN_TARGETS = np.random.randint(0, NUM_CLASSES, size=(NUM_SAMPLES, 1))
VAL_INPUTS = np.random.random(size=(NUM_SAMPLES, INPUT_DIM))
VAL_TARGETS = np.random.randint(0, NUM_CLASSES, size=(NUM_SAMPLES, 1))


@pytest.fixture(scope='module')
def tmp_dir(tmpdir_factory):
    return tmpdir_factory.mktemp('integration_test')


def build_model(hp):
    model = keras.Sequential([
        keras.layers.Dense(hp.Range('units', 100, 1000, 100),
                           input_shape=(INPUT_DIM,),
                           activation='relu'),
        keras.layers.Dense(NUM_CLASSES, activation='softmax')
    ])
    model.compile('rmsprop', 'mse', metrics=['accuracy'])
    return model


def test_tuner_errors():
    # invalid oracle
    with pytest.raises(
            ValueError,
            match='Expected oracle to be an instance of Oracle'):
        tuner_module.Tuner(
            oracle='invalid',
            hypermodel=build_model,
            objective='val_accuracy',
            max_trials=3)
    # invalid hypermodel
    with pytest.raises(
            ValueError,
            match='`hypermodel` argument should be either'):
        tuner_module.Tuner(
            oracle=kerastuner.tuners.randomsearch.RandomSearchOracle(),
            hypermodel='build_model',
            objective='val_accuracy',
            max_trials=3)
    # overside model
    with pytest.raises(
            RuntimeError,
            match='Too many consecutive oversized models'):
        tuner = tuner_module.Tuner(
            oracle=kerastuner.tuners.randomsearch.RandomSearchOracle(),
            hypermodel=build_model,
            objective='val_accuracy',
            max_trials=3,
            max_model_size=4)
        tuner.search(TRAIN_INPUTS, TRAIN_TARGETS,
                     validation_data=(VAL_INPUTS, VAL_TARGETS))
