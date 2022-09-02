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


def test_fixed():
    fixed = hp_module.Fixed("fixed", "value")
    fixed = hp_module.Fixed.from_config(fixed.get_config())
    assert fixed.default == "value"
    assert fixed.random_sample() == "value"

    fixed = hp_module.Fixed("fixed", True)
    assert fixed.default is True
    assert fixed.random_sample() is True

    fixed = hp_module.Fixed("fixed", False)
    fixed = hp_module.Fixed.from_config(fixed.get_config())
    assert fixed.default is False
    assert fixed.random_sample() is False

    fixed = hp_module.Fixed("fixed", 1)
    assert fixed.value == 1
    assert fixed.random_sample() == 1

    fixed = hp_module.Fixed("fixed", 8.2)
    assert fixed.value == 8.2
    assert fixed.random_sample() == 8.2
    assert fixed.value_to_prob(fixed.value) == 0.5

    with pytest.raises(ValueError, match="value must be an"):
        hp_module.Fixed("fixed", None)


def test_fixed_repr():
    assert repr(hp_module.Fixed("fixed", "value")) == repr(
        hp_module.Fixed("fixed", "value")
    )


def test_fixed_values_property():
    assert list(hp_module.Fixed("fixed", 2).values) == [2]
