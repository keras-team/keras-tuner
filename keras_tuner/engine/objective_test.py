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

import keras_tuner
from keras_tuner.engine import objective


def test_create_objective_with_str():
    obj = objective.create_objective("accuracy")
    assert obj.name == "accuracy" and obj.direction == "max"


def test_create_objective_with_objective():
    obj = objective.create_objective("accuracy")
    obj = objective.create_objective(keras_tuner.Objective("score", "min"))
    assert obj.name == "score" and obj.direction == "min"


def test_create_objective_with_multi_objective():
    obj = objective.create_objective(
        [keras_tuner.Objective("score", "max"), keras_tuner.Objective("loss", "min")]
    )
    assert isinstance(obj, objective.MultiObjective)
    assert obj.objectives[0].name == "score" and obj.objectives[0].direction == "max"
    assert obj.objectives[1].name == "loss" and obj.objectives[1].direction == "min"


def test_create_objective_with_multi_str():
    obj = objective.create_objective(["accuracy", "loss"])
    assert isinstance(obj, objective.MultiObjective)
    assert (
        obj.objectives[0].name == "accuracy" and obj.objectives[0].direction == "max"
    )
    assert obj.objectives[1].name == "loss" and obj.objectives[1].direction == "min"


def test_objective_better_than_max():
    obj = objective.create_objective("accuracy")
    assert obj.better_than(1, 0)
    assert not obj.better_than(0, 1)
    assert not obj.better_than(0, 0)


def test_objective_better_than_min():
    obj = objective.create_objective("loss")
    assert obj.better_than(0, 1)
    assert not obj.better_than(1, 0)
    assert not obj.better_than(0, 0)


def test_objective_has_value():
    obj = objective.create_objective("loss")
    assert obj.has_value({"loss": 3.0})
    assert not obj.has_value({"accuracy": 3.0})


def test_objective_get_value():
    obj = objective.create_objective("loss")
    assert obj.get_value({"accuracy": 3.0, "loss": 2.0}) == 2.0


def test_multi_objective_get_value():
    obj = objective.create_objective(["accuracy", "loss"])
    assert obj.get_value({"accuracy": 3.0, "loss": 2.0}) == -1.0


def test_objective_equal():
    obj1 = objective.Objective(name="accuracy", direction="max")
    obj2 = objective.Objective(name="accuracy", direction="max")
    assert obj1 == obj2


def test_objective_not_equal_with_diff_name():
    obj1 = objective.Objective(name="accuracy1", direction="max")
    obj2 = objective.Objective(name="accuracy", direction="max")
    assert obj1 != obj2


def test_objective_not_equal_with_diff_dir():
    obj1 = objective.Objective(name="accuracy", direction="min")
    obj2 = objective.Objective(name="accuracy", direction="max")
    assert obj1 != obj2


def test_multi_objective_equal():
    obj1 = objective.create_objective(["accuracy", "loss"])
    obj2 = objective.create_objective(["loss", "accuracy"])
    assert obj1 == obj2


def test_multi_objective_not_equal():
    obj1 = objective.create_objective(["loss", "loss"])
    obj2 = objective.create_objective(["loss", "accuracy"])
    assert obj1 != obj2


def test_multi_objective_has_value():
    obj = objective.create_objective(["loss", "accuracy"])
    assert obj.has_value({"loss": 1.0, "accuracy": 1.0, "mse": 2.0})
    assert not obj.has_value({"accuracy": 1.0, "mse": 2.0})
