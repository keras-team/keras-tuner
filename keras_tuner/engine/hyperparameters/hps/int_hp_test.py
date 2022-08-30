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

import pytest

from keras_tuner.engine import hyperparameters as hp_module
from keras_tuner.protos import keras_tuner_pb2


def test_int_sampling_arg():
    i = hp_module.Int("i", 0, 10, sampling="linear")
    i = hp_module.Int.from_config(i.get_config())
    assert i.sampling == "linear"

    with pytest.raises(ValueError, match="`sampling` must be one of"):
        hp_module.Int("j", 0, 10, sampling="invalid")

    with pytest.raises(
        ValueError,
        match="`min_value` 1 is greater than the `max_value` 0",
    ):
        hp_module.Int("k", 1, 0, sampling="linear")

    with pytest.raises(
        ValueError,
        match="`min_value` 1 is greater than the `max_value` 0",
    ):
        hp_module.Int("k", 1, 0, sampling="linear")


def test_int():
    rg = hp_module.Int("rg", min_value=5, max_value=9, step=1, default=6)
    rg = hp_module.Int.from_config(rg.get_config())
    assert rg.default == 6
    assert 5 <= rg.random_sample() <= 9
    assert isinstance(rg.random_sample(), int)
    assert rg.random_sample(123) == rg.random_sample(123)
    assert abs(rg.value_to_prob(6) - 0.3) < 1e-4
    # No default
    rg = hp_module.Int("rg", min_value=5, max_value=9, step=1)
    assert rg.default == 5


def test_int_log_with_step():
    rg = hp_module.Int("rg", min_value=2, max_value=32, step=2, sampling="log")
    for i in range(10):
        assert rg.random_sample() in [2, 4, 8, 16, 32]
    assert abs(rg.value_to_prob(4) - 0.3) < 1e-4
    assert rg.prob_to_value(0.3) == 4


def test_int_log_without_step():
    rg = hp_module.Int("rg", min_value=2, max_value=32, sampling="log")
    assert rg.prob_to_value(rg.value_to_prob(4)) == 4


def test_int_proto():
    hp = hp_module.Int("a", 1, 100, sampling="log")
    proto = hp.to_proto()
    assert proto.name == "a"
    assert proto.min_value == 1
    assert proto.max_value == 100
    assert proto.sampling == keras_tuner_pb2.Sampling.LOG
    # Proto stores the implicit default.
    assert proto.default == 1
    assert proto.step == 0

    new_hp = hp_module.Int.from_proto(proto)
    assert new_hp._default == 1
    # Pop the implicit default for comparison purposes.
    new_hp._default = None
    assert new_hp.get_config() == hp.get_config()
