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

import keras_tuner as kt
from keras_tuner.engine import oracle as oracle_module


class OracleStub(oracle_module.Oracle):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.score_trial_called = False

    def populate_space(self, trial_id):
        return "populate_space"

    def score_trial(self, trial_id):
        self.score_trial_called = True


def test_private_populate_space_deprecated_and_call_public():
    oracle = OracleStub(objective="val_loss")
    with pytest.deprecated_call():
        assert oracle._populate_space("100") == "populate_space"


def test_private_score_trial_deprecated_and_call_public():
    oracle = OracleStub(objective="val_loss")
    with pytest.deprecated_call():
        oracle._score_trial("trial")
    assert oracle.score_trial_called


def test_import_objective_from_oracle():
    # This test is for backward compatibility.
    from keras_tuner.engine.oracle import Objective

    assert Objective is kt.Objective
