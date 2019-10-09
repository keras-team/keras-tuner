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
"""Tests for distributed tuning."""

import logging
import os
import numpy as np
import threading
import pytest
import tensorflow as tf
from tensorflow import keras
from unittest import mock

import kerastuner as kt
from kerastuner.distribute import utils as dist_utils
from .. import mock_distribute


@pytest.fixture(scope='function')
def tmp_dir(tmpdir_factory):
    return tmpdir_factory.mktemp('integration_test', numbered=True)


def make_locking_fn(fn, lock):
    def locking_fn(*args, **kwargs):
        with lock:
            return fn(*args, **kwargs)
    return locking_fn


def test_random_search(tmp_dir):
    num_workers = 3
    # TensorFlow model building and execution is not thread-safe.
    lock = threading.RLock()
    barrier = threading.Barrier(num_workers)

    def _test_random_search():
        def build_model(hp):
            model = keras.Sequential()
            for i in range(hp.Int('num_layers', 1, 3)):
                model.add(keras.layers.Dense(
                    hp.Int('num_units_%i' % i, 1, 3),
                    activation='relu'))
            model.add(keras.layers.Dense(1, activation='sigmoid'))
            model.compile('sgd', 'binary_crossentropy')
            return model

        x = np.random.uniform(-1, 1, size=(2, 5))
        y = np.ones((2, 1))

        tuner = kt.tuners.RandomSearch(
            hypermodel=make_locking_fn(build_model, lock),
            objective='val_loss',
            max_trials=3,
            directory=tmp_dir)

        # Only workers make it to this point, server runs until thread stops.
        assert dist_utils.has_chief_oracle()
        assert not dist_utils.is_chief_oracle()
        assert isinstance(tuner.oracle, kt.distribute.oracle_client.OracleClient)

        with mock.patch.object(tuner, 'run_trial', make_locking_fn(tuner.run_trial, lock)):
            # TensorFlow Models are not thread-safe.
            tuner.search(x, y, validation_data=(x, y), epochs=1, batch_size=2)

        barrier.wait(60)
        # Suppress warnings about optimizer state not being restored by tf.keras.
        tf.get_logger().setLevel(logging.ERROR)
        with lock:
            trials = tuner.oracle.get_best_trials(3)
            assert trials[0].score <= trials[1].score
            assert trials[1].score <= trials[2].score

            models = tuner.get_best_models(2)
            #assert round(models[0].evaluate(x, y), 4) <= round(models[1].evaluate(x, y), 4)

    mock_distribute.mock_distribute(_test_random_search, num_workers)
