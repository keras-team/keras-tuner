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
import pytest

from keras_tuner.engine import hyperparameters as hp_module


def test_choice():
    choice = hp_module.Choice("choice", [1, 2, 3], default=2)
    choice = hp_module.Choice.from_config(choice.get_config())
    assert choice.default == 2
    assert choice.random_sample() in [1, 2, 3]
    assert choice.random_sample(123) == choice.random_sample(123)
    assert abs(choice.value_to_prob(1) - 1.0 / 6) < 1e-4
    # No default
    choice = hp_module.Choice("choice", [1, 2, 3])
    assert choice.default == 1
    with pytest.raises(ValueError, match="default value should be"):
        hp_module.Choice("choice", [1, 2, 3], default=4)


@pytest.mark.parametrize(
    "values,ordered_arg,ordered_val",
    [
        ([1, 2, 3], True, True),
        ([1, 2, 3], False, False),
        ([1, 2, 3], None, True),
        (["a", "b", "c"], False, False),
        (["a", "b", "c"], None, False),
    ],
)
def test_choice_ordered(values, ordered_arg, ordered_val):
    choice = hp_module.Choice("choice", values, ordered=ordered_arg)
    assert choice.ordered == ordered_val
    choice_new = hp_module.Choice(**choice.get_config())
    assert choice_new.ordered == ordered_val


def test_choice_ordered_invalid():
    with pytest.raises(ValueError, match="must be `False`"):
        hp_module.Choice("a", ["a", "b"], ordered=True)


def test_choice_types():
    values1 = ["a", "b", 0]
    with pytest.raises(TypeError, match="can contain only one"):
        hp_module.Choice("a", values1)
    values2 = [{"a": 1}, {"a": 2}]
    with pytest.raises(TypeError, match="can contain only `int`"):
        hp_module.Choice("a", values2)


def test_choice_value_not_provided_error():
    with pytest.raises(ValueError, match="`values` must be provided"):
        hp_module.Choice("a", [])


def test_choice_repr():
    assert repr(hp_module.Choice("a", [1, 2, 3])) == repr(
        hp_module.Choice("a", [1, 2, 3])
    )


def test_choice_none_as_default():
    hp = hp_module.Choice("a", [1, 2], default=None)
    assert hp.default == 1


def test_choice_default_not_none():
    hp = hp_module.Choice("a", [1, 2], default=2)
    assert hp.default == 2


def test_choice_proto():
    hp = hp_module.Choice("a", [2.3, 4.5, 6.3], ordered=True)
    proto = hp.to_proto()
    assert proto.name == "a"
    assert proto.ordered
    assert np.allclose([v.float_value for v in proto.values], [2.3, 4.5, 6.3])
    # Proto stores the implicit default.
    assert np.isclose(proto.default.float_value, 2.3)

    new_hp = hp_module.Choice.from_proto(proto)
    assert new_hp.name == "a"
    assert np.allclose(new_hp.values, hp.values)
    assert new_hp.ordered
    assert np.isclose(new_hp._default, 2.3)

    # Test int values.
    int_choice = hp_module.Choice("b", [1, 2, 3], ordered=False, default=2)
    new_int_choice = hp_module.Choice.from_proto(int_choice.to_proto())
    assert int_choice.get_config() == new_int_choice.get_config()

    # Test float values.
    float_choice = hp_module.Choice("b", [0.5, 2.5, 4.0], ordered=False, default=2.5)
    new_float_choice = hp_module.Choice.from_proto(float_choice.to_proto())
    assert float_choice.get_config() == new_float_choice.get_config()


def test_prob_one_choice():
    hp = hp_module.Choice("a", [0, 1, 2])
    # Check that boundaries are valid.
    value = hp.prob_to_value(1)
    assert value == 2

    value = hp.prob_to_value(0)
    assert value == 0


def test_choice_values_property():
    assert list(hp_module.Choice("choice", [0, 1, 2]).values) == [0, 1, 2]
