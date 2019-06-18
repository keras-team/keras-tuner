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

from tensorflow.keras.layers import Input, Dense  # nopep8 pylint: disable=import-error
from tensorflow.keras.models import Model  # pylint: disable=import-error

from kerastuner.engine.tuner import Tuner
from kerastuner.distributions.randomdistributions import RandomDistributions


def test_empty_model_function():
    with pytest.raises(ValueError):
        Tuner(None, 'test', 'loss', RandomDistributions)


def dummy():
    return 1


def test_invalid_model_function():
    with pytest.raises(ValueError):
        Tuner(dummy, 'test', 'loss', RandomDistributions)


def basic_model():
    # Can't pass the model as a fixture, as the tuner expect to call it itself
    i = Input(shape=(1, ), name="input")
    o = Dense(4, name="output")(i)
    model = Model(inputs=i, outputs=o)
    model.compile(optimizer="adam", loss='mse', metrics=['accuracy'])
    return model


def oversized_model():
    # Can't pass the model as a fixture, as the tuner expect to call it itself
    i = Input(shape=(1, ), name="input")
    o = Dense(1024, name="output")(i)
    model = Model(inputs=i, outputs=o)
    model.compile(optimizer="adam", loss='mse', metrics=['accuracy'])
    return model


class TwoTimesOnly(object):
    def __init__(self):
        self.cnt = 0

    def two_time_only_model(self):
        print("Call.")
        if self.cnt < 3:
            self.cnt += 1
            i = Input(shape=(1, ), name="input")
            o = Dense(1024, name="output")(i)
            model = Model(inputs=i, outputs=o)
            model.compile(optimizer="adam", loss='mse', metrics=['accuracy'])
            return model
        return None


class TwoTimesThenThrow(object):
    def __init__(self):
        self.cnt = 0

    def two_time_only_model(self):
        print("Call.")
        if self.cnt < 3:
            self.cnt += 1
            i = Input(shape=(1, ), name="input")
            o = Dense(1024, name="output")(i)
            model = Model(inputs=i, outputs=o)
            model.compile(optimizer="adam", loss='mse', metrics=['accuracy'])
            return model
        raise ValueError('Exception thrown.')


def test_valid_model_function():
    Tuner(basic_model, 'test', 'loss', RandomDistributions)


def test_oversized_models():
    tuner = Tuner(oversized_model,
                  'test',
                  'loss',
                  RandomDistributions,
                  max_model_parameters=10)

    assert tuner.new_instance() == None


def test_exception_in_model_fn():
    o = TwoTimesThenThrow()
    tuner = Tuner(o.two_time_only_model, 'test', 'loss', RandomDistributions)

    # Note that the tuner implicitly calls the model fn once to collect info
    assert tuner.new_instance() is not None
    assert tuner.new_instance() is None


def test_none_in_model_fn():
    o = TwoTimesOnly()
    tuner = Tuner(o.two_time_only_model, 'test', 'loss', RandomDistributions)

    # Note that the tuner implicitly calls the model fn once to collect info
    assert tuner.new_instance() is not None
    assert tuner.new_instance() is None


def test_validation_split_and_data():
    with pytest.raises(ValueError):
        tuner = Tuner(basic_model, 'test', 'loss', RandomDistributions)
        tuner.search(None, None, validation_data="a", validation_split=1)
