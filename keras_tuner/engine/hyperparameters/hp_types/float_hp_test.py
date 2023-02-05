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

import numpy as np

from keras_tuner.engine import hyperparameters as hp_module
from keras_tuner.protos import keras_tuner_pb2


def test_float():
    # Test with step arg
    linear = hp_module.Float(
        "linear", min_value=0.5, max_value=9.5, step=0.1, default=9.0
    )
    linear = hp_module.Float.from_config(linear.get_config())
    assert linear.default == 9.0
    assert 0.5 <= linear.random_sample() <= 9.5
    assert isinstance(linear.random_sample(), float)
    assert linear.random_sample(123) == linear.random_sample(123)

    # Test without step arg
    linear = hp_module.Float("linear", min_value=0.5, max_value=6.5, default=2.0)
    linear = hp_module.Float.from_config(linear.get_config())
    assert linear.default == 2.0
    assert 0.5 <= linear.random_sample() < 6.5
    assert isinstance(linear.random_sample(), float)
    assert linear.random_sample(123) == linear.random_sample(123)

    # No default
    linear = hp_module.Float("linear", min_value=0.5, max_value=9.5, step=0.1)
    assert linear.default == 0.5


def test_float_linear_value_to_prob_no_step():
    rg = hp_module.Float("rg", min_value=1.0, max_value=11.0)
    assert abs(rg.value_to_prob(4.5) - 0.35) < 1e-4
    assert rg.prob_to_value(0.35) == 4.5


def test_float_log_with_step():
    rg = hp_module.Float(
        "rg", min_value=0.01, max_value=100, step=10, sampling="log"
    )
    for _ in range(10):
        assert rg.random_sample() in [0.01, 0.1, 1.0, 10.0, 100.0]
    assert abs(rg.value_to_prob(0.1) - 0.3) < 1e-4
    assert rg.prob_to_value(0.3) == 0.1


def test_float_reverse_log_with_step():
    rg = hp_module.Float(
        "rg", min_value=0.01, max_value=100, step=10, sampling="reverse_log"
    )
    for _ in range(10):
        # print(rg.random_sample())
        # assert rg.random_sample() in [0.01, 0.1, 1.0, 10.0, 100.0]
        # [0.09, 0.9, 9, 90]
        # [90, 9, 0.9, 0.09]
        sample = rg.random_sample()
        assert any(
            abs(sample - x) < 1e-4 for x in [0.01, 90.01, 99.01, 99.91, 100.0]
        )
    assert abs(rg.value_to_prob(99.91) - 0.3) < 1e-4
    assert abs(rg.prob_to_value(0.3) - 99.91) < 1e-4


def test_sampling_zero_length_intervals():
    f = hp_module.Float("f", 2, 2)
    rand_sample = f.random_sample()
    assert rand_sample == 2

    val = 2
    prob = f.value_to_prob(val)
    assert prob == 0.5


def test_log_sampling_random_state():
    f = hp_module.Float("f", 1e-3, 1e3, sampling="log")
    rand_sample = f.random_sample()
    assert rand_sample >= f.min_value
    assert rand_sample <= f.max_value

    val = 1e-3
    prob = f.value_to_prob(val)
    assert prob == 0
    new_val = f.prob_to_value(prob)
    assert np.isclose(val, new_val)

    val = 1
    prob = f.value_to_prob(val)
    assert prob == 0.5
    new_val = f.prob_to_value(prob)
    assert np.isclose(val, new_val)

    val = 1e3
    prob = f.value_to_prob(val)
    assert prob == 1
    new_val = f.prob_to_value(prob)
    assert np.isclose(val, new_val)


def test_reverse_log_sampling_random_state():
    f = hp_module.Float("f", 1e-3, 1e3, sampling="reverse_log")
    rand_sample = f.random_sample()
    assert rand_sample >= f.min_value
    assert rand_sample <= f.max_value

    val = 1e-3
    prob = f.value_to_prob(val)
    assert prob == 0
    new_val = f.prob_to_value(prob)
    assert np.isclose(val, new_val)

    val = 1
    prob = f.value_to_prob(val)
    assert prob > 0 and prob < 1
    new_val = f.prob_to_value(prob)
    assert np.isclose(val, new_val)


def test_float_sampling_arg():
    f = hp_module.Float("f", 1e-20, 1e10, sampling="log")
    f = hp_module.Float.from_config(f.get_config())
    assert f.sampling == "log"


def test_float_proto():
    hp = hp_module.Float("a", -10, 10, sampling="linear", default=3)
    proto = hp.to_proto()
    assert proto.name == "a"
    assert proto.min_value == -10.0
    assert proto.max_value == 10.0
    assert proto.sampling == keras_tuner_pb2.Sampling.LINEAR
    assert proto.default == 3.0
    # Zero is the default, gets converted to `None` in `from_proto`.
    assert proto.step == 0.0

    new_hp = hp_module.Float.from_proto(proto)
    assert new_hp.get_config() == hp.get_config()


def test_float_values_property_with_step():
    assert list(hp_module.Float("float", 2, 8, 2).values) == [2.0, 4.0, 6.0, 8.0]
    assert isinstance(list(hp_module.Float("float", 2, 8, 2).values)[0], float)
    assert list(hp_module.Float("float", 0.1, 100.0, 10, sampling="log").values) == [
        0.1,
        1.0,
        10.0,
        100.0,
    ]


def test_float_values_property_without_step():
    assert len(list(hp_module.Float("float", 2, 4).values)) == 10
    assert len(list(hp_module.Float("float", 2, 20).values)) == 10
    assert len(list(hp_module.Float("float", 2, 1024, sampling="log").values)) == 10
