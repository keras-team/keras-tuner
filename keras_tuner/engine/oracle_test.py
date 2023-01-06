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

import keras_tuner
from keras_tuner.engine import oracle as oracle_module
from keras_tuner.engine import trial as trial_module


class OracleStub(oracle_module.Oracle):
    def __init__(self, directory, **kwargs):
        super().__init__(**kwargs)
        self.score_trial_called = False
        self._set_project_dir(
            directory=directory, project_name="name", overwrite=True
        )

    def populate_space(self, trial_id):
        return {
            "values": {"hp1": "populate_space"},
            "status": trial_module.TrialStatus.RUNNING,
        }

    def score_trial(self, trial_id):
        super().score_trial(trial_id)
        self.score_trial_called = True


def test_private_populate_space_deprecated_and_call_public(tmp_path):
    oracle = OracleStub(directory=tmp_path, objective="val_loss")
    with pytest.deprecated_call():
        assert isinstance(oracle._populate_space("100"), dict)


def test_private_score_trial_deprecated_and_call_public(tmp_path):
    oracle = OracleStub(directory=tmp_path, objective="val_loss")
    trial = oracle.create_trial(tuner_id="a")
    oracle.update_trial(trial_id=trial.trial_id, metrics={"val_loss": 0.5})
    with pytest.deprecated_call():
        oracle._score_trial(trial)
    assert oracle.score_trial_called


def test_import_objective_from_oracle():
    # This test is for backward compatibility.
    from keras_tuner.engine.oracle import Objective

    assert Objective is keras_tuner.Objective


def test_duplicate(tmp_path):
    class MyOracle(OracleStub):
        def populate_space(self, trial_id):
            values = {"hp1": 1}
            if len(self.ongoing_trials) > 0:
                assert self._duplicate(values)
            return {
                "values": values,
                "status": trial_module.TrialStatus.RUNNING,
            }

    oracle = MyOracle(directory=tmp_path, objective="val_loss")
    oracle.create_trial(tuner_id="a")
    oracle.create_trial(tuner_id="b")
    assert len(oracle.ongoing_trials) == 2


def test_not_duplicate(tmp_path):
    class MyOracle(OracleStub):
        def populate_space(self, trial_id):
            values = {"hp1": len(self.ongoing_trials)}
            assert not self._duplicate(values)
            return {
                "values": values,
                "status": trial_module.TrialStatus.RUNNING,
            }

    oracle = MyOracle(directory=tmp_path, objective="val_loss")
    oracle.create_trial(tuner_id="a")
    oracle.create_trial(tuner_id="b")
    assert len(oracle.ongoing_trials) == 2


def test_new_hp_duplicate(tmp_path):
    class MyOracle(OracleStub):
        def populate_space(self, trial_id):
            values = {"hp1": 1}
            assert not self._duplicate(values)
            if len(self.end_order) > 0:
                values["hp2"] = 2
                assert self._duplicate(values)
            return {
                "values": values,
                "status": trial_module.TrialStatus.RUNNING,
            }

    oracle = MyOracle(directory=tmp_path, objective="val_loss")
    trial = oracle.create_trial(tuner_id="a")
    trial.hyperparameters.values["hp2"] = 2
    oracle.update_trial(trial.trial_id, {"val_loss": 3.0})
    oracle.end_trial(trial)
    oracle.create_trial(tuner_id="b")
    assert len(oracle.start_order) == 2


def test_default_no_retry(tmp_path):
    oracle = OracleStub(directory=tmp_path, objective="val_loss")
    trial_1 = oracle.create_trial(tuner_id="a")
    trial_1.status = trial_module.TrialStatus.INVALID
    trial_1.message = "error1"
    oracle.end_trial(trial_1)

    trial_2 = oracle.create_trial(tuner_id="a")
    assert trial_1.trial_id != trial_2.trial_id


def test_retry_invalid_trial(tmp_path):
    oracle = OracleStub(
        directory=tmp_path, objective="val_loss", max_retries_per_trial=1
    )
    trial_1 = oracle.create_trial(tuner_id="a")
    trial_1.status = trial_module.TrialStatus.INVALID
    trial_1.message = "error1"
    oracle.end_trial(trial_1)

    # This is the retry for the trial.
    trial_2 = oracle.create_trial(tuner_id="a")
    assert trial_1.trial_id == trial_2.trial_id

    # Retried once. This is a new trial.
    trial_3 = oracle.create_trial(tuner_id="b")
    assert trial_1.trial_id != trial_3.trial_id


def test_is_nan_mark_as_invalid(tmp_path):
    oracle = OracleStub(
        directory=tmp_path, objective="val_loss", max_retries_per_trial=1
    )
    trial = oracle.create_trial(tuner_id="a")
    oracle.update_trial(trial.trial_id, metrics={"val_loss": float("nan")})
    trial.status = trial_module.TrialStatus.COMPLETED
    trial.message = "error1"
    oracle.end_trial(trial)
    assert oracle.trials[trial.trial_id].status == trial_module.TrialStatus.INVALID


def test_no_retry_for_failed_trial(tmp_path):
    oracle = OracleStub(
        directory=tmp_path, objective="val_loss", max_retries_per_trial=1
    )
    trial_1 = oracle.create_trial(tuner_id="a")
    # Failed, so no retry.
    trial_1.status = trial_module.TrialStatus.FAILED
    trial_1.message = "error1"
    oracle.end_trial(trial_1)

    trial_2 = oracle.create_trial(tuner_id="a")
    assert trial_1.trial_id != trial_2.trial_id


def test_consecutive_failures_in_limit(tmp_path):
    oracle = OracleStub(
        directory=tmp_path, objective="val_loss", max_retries_per_trial=2
    )

    # (1 run + 2 retry) * 2 trial = 6
    for _ in range(6):
        trial = oracle.create_trial(tuner_id="a")
        # Failed, so no retry.
        trial.status = trial_module.TrialStatus.INVALID
        trial.message = "error1"
        oracle.end_trial(trial)

    for _ in range(3):
        trial = oracle.create_trial(tuner_id="a")
        # Failed, so no retry.
        trial.status = trial_module.TrialStatus.COMPLETED
        oracle.update_trial(trial.trial_id, metrics={"val_loss": 0.5})
        oracle.end_trial(trial)


def test_too_many_consecutive_failures(tmp_path):
    oracle = OracleStub(
        directory=tmp_path, objective="val_loss", max_retries_per_trial=2
    )

    with pytest.raises(RuntimeError, match="Number of consecutive") as e:
        for _ in range(3):
            trial = oracle.create_trial(tuner_id="a")
            # Failed, so no retry.
            trial.status = trial_module.TrialStatus.FAILED
            trial.message = "custom_error_info"
            oracle.end_trial(trial)
        assert "custom_error_info" in str(e)
