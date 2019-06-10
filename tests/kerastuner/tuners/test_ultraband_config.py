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
from kerastuner.tuners.ultraband.ultraband_config import UltraBandConfig


def test_epoch_sequence():
    ubc = UltraBandConfig(3, 3, 27, 1000)
    assert np.array_equal(ubc.epoch_sequence, [3, 9, 27])
    assert np.array_equal(ubc.model_sequence, [27, 9, 3])

    ubc = UltraBandConfig(3, 4, 27, 1000)
    assert np.array_equal(ubc.epoch_sequence, [4, 12, 27])
    assert np.array_equal(ubc.model_sequence, [27, 12, 4])

    ubc = UltraBandConfig(4, 3, 27, 1000)
    assert np.array_equal(ubc.epoch_sequence, [3, 12, 27])
    assert np.array_equal(ubc.model_sequence, [27, 12, 3])


def test_delta_epoch_sequence():
    ubc = UltraBandConfig(3, 3, 27, 1000)
    assert np.array_equal(ubc.delta_epoch_sequence, [3, 6, 18])

    ubc = UltraBandConfig(3, 4, 27, 1000)
    assert np.array_equal(ubc.delta_epoch_sequence, [4, 8, 15])

    ubc = UltraBandConfig(4, 3, 27, 1000)
    assert np.array_equal(ubc.delta_epoch_sequence, [3, 9, 15])


def test_num_brackets():
    ubc = UltraBandConfig(3, 3, 27, 1000)
    assert ubc.num_brackets == 3

    ubc = UltraBandConfig(3, 3, 28, 1000)
    assert ubc.num_brackets == 4

    ubc = UltraBandConfig(3, 3, 27, 1000)
    assert ubc.num_brackets == 3

    ubc = UltraBandConfig(3, 4, 36, 1000)
    assert ubc.num_brackets == 3

    ubc = UltraBandConfig(3, 4, 37, 1000)
    assert ubc.num_brackets == 4

    ubc = UltraBandConfig(3, 4, 38, 1000)
    assert ubc.num_brackets == 4


def test_total_epochs_per_band():
    ubc = UltraBandConfig(3, 3, 27, 1000)
    assert ubc.total_epochs_per_band == 189

    ubc = UltraBandConfig(3, 3, 28, 1000)
    assert ubc.total_epochs_per_band == 411

    ubc = UltraBandConfig(3, 4, 36, 1000)
    assert ubc.total_epochs_per_band == 336

    ubc = UltraBandConfig(3, 4, 37, 1000)
    assert ubc.total_epochs_per_band == 728

    ubc = UltraBandConfig(3, 4, 38, 1000)
    assert ubc.total_epochs_per_band == 736
    

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
