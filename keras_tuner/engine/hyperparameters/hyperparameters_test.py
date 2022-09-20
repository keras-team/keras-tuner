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
from tensorflow import keras

from keras_tuner.engine import hyperparameters as hp_module
from keras_tuner.protos import keras_tuner_pb2


def test_hyperparameters():
    hp = hp_module.HyperParameters()
    assert hp.values == {}
    assert hp.space == []
    hp.Choice("choice", [1, 2, 3], default=2)
    assert hp.values == {"choice": 2}
    assert len(hp.space) == 1
    assert hp.space[0].name == "choice"
    hp.values["choice"] = 3
    assert hp.get("choice") == 3
    hp = hp.copy()
    assert hp.values == {"choice": 3}
    assert len(hp.space) == 1
    assert hp.space[0].name == "choice"
    with pytest.raises(KeyError, match="does not exist"):
        hp.get("wrong")


def test_name_collision():
    # TODO: figure out how name collision checks
    # should work.
    pass


def test_name_scope():
    hp = hp_module.HyperParameters()
    hp.Choice("choice", [1, 2, 3], default=2)
    with hp.name_scope("scope1"):
        hp.Choice("choice", [4, 5, 6], default=5)
        with hp.name_scope("scope2"):
            hp.Choice("choice", [7, 8, 9], default=8)
        hp.Int("range", min_value=0, max_value=10, step=1, default=0)
    assert hp.values == {
        "choice": 2,
        "scope1/choice": 5,
        "scope1/scope2/choice": 8,
        "scope1/range": 0,
    }


def test_parent_name():
    hp = hp_module.HyperParameters()
    hp.Choice("a", [1, 2, 3], default=2)
    b1 = hp.Int("b", 0, 10, parent_name="a", parent_values=1, default=5)
    b2 = hp.Int("b", 0, 100, parent_name="a", parent_values=2, default=4)
    assert b1 is None
    assert b2 == 4
    # Only active values appear in `values`.
    assert hp.values == {"a": 2, "b": 4}


def test_conditional_scope():
    hp = hp_module.HyperParameters()
    hp.Choice("choice", [1, 2, 3], default=2)
    # Assignment to a non-active conditional hyperparameter returns `None`.
    with hp.conditional_scope("choice", [1, 3]):
        child1 = hp.Choice("child_choice", [4, 5, 6])
    assert child1 is None
    # Retrieve a non-active hp, still none.
    with hp.conditional_scope("choice", [1, 3]):
        child1 = hp.Choice("child_choice", [4, 5, 6])
    assert child1 is None
    # Assignment to an active conditional hyperparameter returns the value.
    with hp.conditional_scope("choice", 2):
        child2 = hp.Choice("child_choice", [7, 8, 9])
    assert child2 == 7
    # Retrieve the value, still same value.
    with hp.conditional_scope("choice", 2):
        child2 = hp.Choice("child_choice", [7, 8, 9])
    assert child2 == 7
    # Only active values appear in `values`.
    assert hp.values == {"choice": 2, "child_choice": 7}

    with pytest.raises(ValueError, match="not defined"):
        with hp.conditional_scope("not_defined_hp", 2):
            hp.Choice("child_choice", [7, 8, 9])


def test_to_proto_unrecognized_hp_type():
    hps = hp_module.HyperParameters()
    hps._space.append(None)
    hps.Fixed("d", "3")

    with pytest.raises(ValueError, match="Unrecognized HP"):
        hp_module.HyperParameters.from_proto(hps.to_proto())


def test_to_proto_unrecognized_value_type():
    hps = hp_module.HyperParameters()
    hps.Fixed("d", "3")
    hps.values["d"] = None

    with pytest.raises(ValueError, match="Unrecognized value type"):
        hp_module.HyperParameters.from_proto(hps.to_proto())


def test_is_active_with_hp_name_and_hp():
    hp = hp_module.HyperParameters()
    hp.Choice("choice", [1, 2, 3], default=3)
    with hp.conditional_scope("choice", [1, 3]):
        hp.Choice("child_choice", [4, 5, 6])
    with hp.conditional_scope("choice", 2):
        hp.Choice("child_choice2", [7, 8, 9])

    # Custom oracle populates value for an inactive hp.
    hp.values["child_choice2"] = 7

    assert hp.is_active("child_choice")
    assert hp.is_active(hp._hps["child_choice"][0])

    assert not hp.is_active("child_choice2")
    assert not hp.is_active(hp._hps["child_choice2"][0])


def test_build_with_conditional_scope():
    def build_model(hp):
        model = hp.Choice("model", ["v1", "v2"])
        with hp.conditional_scope("model", "v1"):
            v1_params = {
                "layers": hp.Int("layers", 1, 3),
                "units": hp.Int("units", 16, 32),
            }
        with hp.conditional_scope("model", "v2"):
            v2_params = {
                "layers": hp.Int("layers", 2, 4),
                "units": hp.Int("units", 32, 64),
            }

        params = v1_params if model == "v1" else v2_params
        inputs = keras.Input(10)
        x = inputs
        for _ in range(params["layers"]):
            x = keras.layers.Dense(params["units"])(x)
        outputs = keras.layers.Dense(1)(x)
        model = keras.Model(inputs, outputs)
        model.compile("sgd", "mse")
        return model

    hp = hp_module.HyperParameters()
    build_model(hp)
    assert hp.values == {
        "model": "v1",
        "layers": 1,
        "units": 16,
    }


def test_error_when_hp_same_name_as_condition():
    hp = hp_module.HyperParameters()
    hp.Choice("a", [1, 2, 3], default=3)
    with pytest.raises(ValueError, match="cannot have the same name"):
        with hp.conditional_scope("a", [1, 3]):
            hp.Choice("a", [4, 5, 6], default=6)


def test_nested_conditional_scopes_and_name_scopes():
    hp = hp_module.HyperParameters()
    a = hp.Choice("a", [1, 2, 3], default=3)
    with hp.conditional_scope("a", [1, 3]):
        b = hp.Choice("b", [4, 5, 6], default=6)
        with hp.conditional_scope("b", 6):
            c = hp.Choice("c", [7, 8, 9])
            with hp.name_scope("d"):
                e = hp.Choice("e", [10, 11, 12])
    with hp.conditional_scope("a", 2):
        f = hp.Choice("f", [13, 14, 15])
        with hp.name_scope("g"):
            h = hp.Int("h", 0, 10)

    assert hp.values == {
        "a": 3,
        "b": 6,
        "c": 7,
        "d/e": 10,
    }
    # Assignment to an active conditional hyperparameter returns the value.
    assert a == 3
    assert b == 6
    assert c == 7
    assert e == 10
    # Assignment to a non-active conditional hyperparameter returns `None`.
    assert f is None
    assert h is None


def test_get_with_conditional_scopes():
    hp = hp_module.HyperParameters()
    hp.Choice("a", [1, 2, 3], default=2)
    assert hp.get("a") == 2
    with hp.conditional_scope("a", 2):
        hp.Fixed("b", 4)
        assert hp.get("b") == 4
        assert hp.get("a") == 2
    with hp.conditional_scope("a", 3):
        hp.Fixed("b", 5)
        assert hp.get("b") == 4

    # Value corresponding to the currently active condition is returned.
    assert hp.get("b") == 4


def test_merge_inactive_hp_with_conditional_scopes():
    hp = hp_module.HyperParameters()
    hp.Choice("a", [1, 2, 3], default=3)
    assert hp.get("a") == 3
    with hp.conditional_scope("a", 2):
        hp.Fixed("b", 4)

    hp2 = hp_module.HyperParameters()
    hp2.merge(hp)
    # only active hp should be included to values
    assert "a" in hp2.values
    assert "b" not in hp2.values


def test_merge():
    hp = hp_module.HyperParameters()
    hp.Int("a", 0, 100)
    hp.Fixed("b", 2)

    hp2 = hp_module.HyperParameters()
    hp2.Fixed("a", 3)
    hp.Int("c", 10, 100, default=30)

    hp.merge(hp2)

    assert hp.get("a") == 3
    assert hp.get("b") == 2
    assert hp.get("c") == 30

    hp3 = hp_module.HyperParameters()
    hp3.Fixed("a", 5)
    hp3.Choice("d", [1, 2, 3], default=1)

    hp.merge(hp3, overwrite=False)

    assert hp.get("a") == 3
    assert hp.get("b") == 2
    assert hp.get("c") == 30
    assert hp.get("d") == 1


def _sort_space(hps):
    space = hps.get_config()["space"]
    return sorted(space, key=lambda hp: hp["config"]["name"])


def test_hyperparameters_proto():
    hps = hp_module.HyperParameters()
    hps.Int("a", 1, 10, sampling="reverse_log", default=3)
    hps.Float("b", 2, 8, sampling="linear", default=4)
    hps.Choice("c", [1, 5, 10], ordered=False, default=5)
    hps.Fixed("d", "3")
    hps.Fixed("e", 3)
    hps.Fixed("f", 3.1)
    hps.Fixed("g", True)
    hps.Boolean("h")
    with hps.name_scope("d"):
        hps.Choice("e", [2.0, 4.5, 8.5], default=2.0)
        hps.Choice("f", ["1", "2"], default="1")
        with hps.conditional_scope("f", "1"):
            hps.Int("g", -10, 10, step=2, default=-2)

    new_hps = hp_module.HyperParameters.from_proto(hps.to_proto())
    assert _sort_space(hps) == _sort_space(new_hps)
    assert hps.values == new_hps.values


def test_hyperparameters_values_proto():
    values = keras_tuner_pb2.HyperParameters.Values(
        values={
            "a": keras_tuner_pb2.Value(int_value=1),
            "b": keras_tuner_pb2.Value(float_value=2.0),
            "c": keras_tuner_pb2.Value(string_value="3"),
        }
    )

    # When only values are provided, each param is created as `Fixed`.
    hps = hp_module.HyperParameters.from_proto(values)
    assert hps.values == {"a": 1, "b": 2.0, "c": "3"}


def test_dict_methods():
    hps = hp_module.HyperParameters()
    hps.Int("a", 0, 10, default=3)
    hps.Choice("b", [1, 2], default=2)
    with hps.conditional_scope("b", 1):
        hps.Float("c", -10, 10, default=3)
        # Don't allow access of a non-active param within its scope.
        with pytest.raises(ValueError, match="is currently inactive"):
            hps["c"]
    with hps.conditional_scope("b", 2):
        hps.Float("c", -30, -20, default=-25)

    assert hps["a"] == 3
    assert hps["b"] == 2
    # Ok to access 'c' here since there is an active 'c'.
    assert hps["c"] == -25
    with pytest.raises(KeyError, match="does not exist"):
        hps["d"]

    assert "a" in hps
    assert "b" in hps
    assert "c" in hps
    assert "d" not in hps


def test_return_populated_value_for_new_hp():
    hp = hp_module.HyperParameters()

    hp.values["hp_name"] = "hp_value"
    assert (
        hp.Choice(
            "hp_name", ["hp_value", "hp_value_default"], default="hp_value_default"
        )
        == "hp_value"
    )


def test_return_default_value_if_not_populated():
    hp = hp_module.HyperParameters()

    assert (
        hp.Choice(
            "hp_name", ["hp_value", "hp_value_default"], default="hp_value_default"
        )
        == "hp_value_default"
    )


def test_serialize_deserialize_hyperparameters():
    hp = hp_module.HyperParameters()
    hp.Int("temp", 1, 5)
    hp = hp_module.deserialize(hp_module.serialize(hp))
    assert len(hp.space) == 1


def test_int_log_without_step_random_sample():
    hp = hp_module.HyperParameters()
    hp.Int("rg", min_value=2, max_value=32, sampling="log")
    hp.space[0].random_sample()
