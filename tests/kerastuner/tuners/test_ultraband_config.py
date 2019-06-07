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
from kerastuner.tuners.ultraband import UltraBandConfig


def test_epoch_sequence():
    ubc = UltraBandConfig(3, 3, 27, 1000)
    assert np.array_equal(ubc.epoch_sequence, [3, 6, 18])
    assert np.array_equal(ubc.model_sequence, [27, 9, 3])

    ubc = UltraBandConfig(3, 4, 27, 1000)
    assert np.array_equal(ubc.epoch_sequence, [4, 8, 15])
    assert np.array_equal(ubc.model_sequence, [27, 12, 4])

    ubc = UltraBandConfig(4, 3, 27, 1000)
    assert np.array_equal(ubc.epoch_sequence, [3, 9, 15])
    assert np.array_equal(ubc.model_sequence, [27, 12, 3])


def test_num_bands():
    ubc = UltraBandConfig(3, 3, 27, 1000)
    assert ubc.num_bands == 3

    ubc = UltraBandConfig(3, 3, 28, 1000)
    assert ubc.num_bands == 4

    ubc = UltraBandConfig(3, 3, 27, 1000)
    assert ubc.num_bands == 3

    ubc = UltraBandConfig(3, 4, 36, 1000)
    assert ubc.num_bands == 3

    ubc = UltraBandConfig(3, 4, 37, 1000)
    assert ubc.num_bands == 4

    ubc = UltraBandConfig(3, 4, 38, 1000)
    assert ubc.num_bands == 4


def test_epochs_per_batch():
    ubc = UltraBandConfig(3, 3, 27, 1000)
    assert ubc.epochs_per_batch == 189

    ubc = UltraBandConfig(3, 3, 28, 1000)
    assert ubc.epochs_per_batch == 411

    ubc = UltraBandConfig(3, 4, 36, 1000)
    assert ubc.epochs_per_batch == 336

    ubc = UltraBandConfig(3, 4, 37, 1000)
    assert ubc.epochs_per_batch == 728

    ubc = UltraBandConfig(3, 4, 38, 1000)
    assert ubc.epochs_per_batch == 736


def test_num_batches():
    ubc = UltraBandConfig(3, 3, 27, 1000)
    assert int(ubc.num_batches) == 5

    ubc = UltraBandConfig(3, 3, 28, 1000)
    assert int(ubc.num_batches) == 2

    ubc = UltraBandConfig(3, 4, 35, 1000)
    assert int(ubc.num_batches) == 3

    ubc = UltraBandConfig(3, 4, 36, 1000)
    assert int(ubc.num_batches) == 2

    ubc = UltraBandConfig(3, 4, 37, 1000)
    assert int(ubc.num_batches) == 1

    ubc = UltraBandConfig(3, 4, 38, 1000)
    assert int(ubc.num_batches) == 1
