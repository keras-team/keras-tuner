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

from unittest import mock

import pytest

from keras_tuner.engine import conditions


def test_from_proto_error():
    with pytest.raises(ValueError, match="Unrecognized condition"):
        proto = mock.Mock()
        proto.WhichOneof.return_value = "unknown"
        conditions.Condition.from_proto(proto)


def test_unknown_condition_values_error():
    with pytest.raises(TypeError, match="Can contain only"):
        conditions.Parent("parent", [None])


def test_int_to_from_proto():
    condition = conditions.Parent("parent", [1, 2])
    condition_2 = conditions.Condition.from_proto(condition.to_proto())
    assert condition.values == condition_2.values


def test_float_to_from_proto():
    condition = conditions.Parent("parent", [1.0, 2.0])
    condition_2 = conditions.Condition.from_proto(condition.to_proto())
    assert condition.values == condition_2.values


def test_bool_to_from_proto():
    condition = conditions.Parent("parent", [True, False])
    condition_2 = conditions.Condition.from_proto(condition.to_proto())
    assert condition.values == condition_2.values
