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
"""Tests for the OracleServicer class."""

import os

import keras_tuner
from keras_tuner.distribute import oracle_chief
from keras_tuner.distribute import oracle_client
from keras_tuner.engine import metrics_tracking
from keras_tuner.test_utils import mock_distribute
from keras_tuner.tuners import randomsearch


def test_get_space(tmp_path):
    def _test_get_space():
        hps = keras_tuner.HyperParameters()
        hps.Int("a", 0, 10, default=3)
        oracle = randomsearch.RandomSearchOracle(
            objective=keras_tuner.Objective("score", "max"),
            max_trials=10,
            hyperparameters=hps,
        )
        oracle._set_project_dir(tmp_path, "untitled")
        tuner_id = os.environ["KERASTUNER_TUNER_ID"]
        if "chief" in tuner_id:
            oracle_chief.start_server(oracle)
        else:
            client = oracle_client.OracleClient(oracle)
            retrieved_hps = client.get_space()
            assert retrieved_hps.values == {"a": 3}
            assert len(retrieved_hps.space) == 1

    mock_distribute.mock_distribute(_test_get_space)


def test_update_space(tmp_path):
    def _test_update_space():
        oracle = randomsearch.RandomSearchOracle(
            objective=keras_tuner.Objective("score", "max"), max_trials=10
        )
        oracle._set_project_dir(tmp_path, "untitled")
        tuner_id = os.environ["KERASTUNER_TUNER_ID"]
        if "chief" in tuner_id:
            oracle_chief.start_server(oracle)
        else:
            client = oracle_client.OracleClient(oracle)

            hps = keras_tuner.HyperParameters()
            hps.Int("a", 0, 10, default=5)
            hps.Choice("b", [1, 2, 3])
            client.update_space(hps)

            retrieved_hps = client.get_space()
            assert len(retrieved_hps.space) == 2
            assert retrieved_hps.values["a"] == 5
            assert retrieved_hps.values["b"] == 1

    mock_distribute.mock_distribute(_test_update_space)


def test_create_trial(tmp_path):
    def _test_create_trial():
        hps = keras_tuner.HyperParameters()
        hps.Int("a", 0, 10, default=5)
        hps.Choice("b", [1, 2, 3])
        oracle = randomsearch.RandomSearchOracle(
            objective=keras_tuner.Objective("score", "max"),
            max_trials=10,
            hyperparameters=hps,
        )
        oracle._set_project_dir(tmp_path, "untitled")
        tuner_id = os.environ["KERASTUNER_TUNER_ID"]
        if "chief" in tuner_id:
            oracle_chief.start_server(oracle)
        else:
            client = oracle_client.OracleClient(oracle)
            trial = client.create_trial(tuner_id)
            assert trial.status == "RUNNING"
            a = trial.hyperparameters.get("a")
            assert a >= 0 and a <= 10
            b = trial.hyperparameters.get("b")
            assert b in {1, 2, 3}

    mock_distribute.mock_distribute(_test_create_trial)


def test_update_trial(tmp_path):
    def _test_update_trial():
        hps = keras_tuner.HyperParameters()
        hps.Int("a", 0, 10, default=5)
        oracle = randomsearch.RandomSearchOracle(
            objective=keras_tuner.Objective("score", "max"),
            max_trials=10,
            hyperparameters=hps,
        )
        oracle._set_project_dir(tmp_path, "untitled")
        tuner_id = os.environ["KERASTUNER_TUNER_ID"]
        if "chief" in tuner_id:
            oracle_chief.start_server(oracle)
        else:
            client = oracle_client.OracleClient(oracle)
            trial = client.create_trial(tuner_id)
            assert "score" not in trial.metrics.metrics
            trial_id = trial.trial_id
            client.update_trial(trial_id, {"score": 1}, step=2)
            updated_trial = client.get_trial(trial_id)
            assert updated_trial.metrics.get_history("score") == [
                metrics_tracking.MetricObservation([1], step=2)
            ]

    mock_distribute.mock_distribute(_test_update_trial)


def test_end_trial(tmp_path):
    def _test_end_trial():
        hps = keras_tuner.HyperParameters()
        hps.Int("a", 0, 10, default=5)
        oracle = randomsearch.RandomSearchOracle(
            objective=keras_tuner.Objective("score", "max"),
            max_trials=10,
            hyperparameters=hps,
        )
        oracle._set_project_dir(tmp_path, "untitled")
        tuner_id = os.environ["KERASTUNER_TUNER_ID"]
        if "chief" in tuner_id:
            oracle_chief.start_server(oracle)
        else:
            client = oracle_client.OracleClient(oracle)
            trial = client.create_trial(tuner_id)
            trial_id = trial.trial_id
            client.update_trial(trial_id, {"score": 1}, step=2)
            trial.status = "FAILED"
            client.end_trial(trial)
            updated_trial = client.get_trial(trial_id)
            assert updated_trial.status == "FAILED"

    mock_distribute.mock_distribute(_test_end_trial)


def test_get_best_trials(tmp_path):
    def _test_get_best_trials():
        hps = keras_tuner.HyperParameters()
        hps.Int("a", 0, 100, default=5)
        hps.Int("b", 0, 100, default=6)
        oracle = randomsearch.RandomSearchOracle(
            objective=keras_tuner.Objective("score", direction="max"),
            max_trials=10,
            hyperparameters=hps,
        )
        oracle._set_project_dir(tmp_path, "untitled")
        tuner_id = os.environ["KERASTUNER_TUNER_ID"]
        if "chief" in tuner_id:
            oracle_chief.start_server(oracle)
        else:
            client = oracle_client.OracleClient(oracle)
            trial_scores = {}
            for score in range(10):
                trial = client.create_trial(tuner_id)
                assert trial.status == "RUNNING"
                assert "a" in trial.hyperparameters.values
                assert "b" in trial.hyperparameters.values
                trial_id = trial.trial_id
                client.update_trial(trial_id, {"score": score})
                trial.status = "COMPLETED"
                client.end_trial(trial)
                trial_scores[trial_id] = score
            best_trials = client.get_best_trials(3)
            best_scores = [t.score for t in best_trials]
            assert best_scores == [9, 8, 7]
            # Check that trial_ids are correctly mapped to scores.
            for t in best_trials:
                assert trial_scores[t.trial_id] == t.score

    mock_distribute.mock_distribute(_test_get_best_trials, num_workers=1)
