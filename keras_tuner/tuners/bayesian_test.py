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

import math

import numpy as np
import pytest
import tensorflow as tf

import keras_tuner
from keras_tuner.engine import hyperparameters as hp_module
from keras_tuner.engine import trial as trial_module
from keras_tuner.tuners import bayesian as bo_module


def build_model(hp):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=(2, 2)))
    for i in range(3):
        model.add(
            tf.keras.layers.Dense(
                units=hp.Int(f"units_{str(i)}", 2, 4, 2), activation="relu"
            )
        )

    model.add(tf.keras.layers.Dense(2, activation="softmax"))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            hp.Choice("learning_rate", [1e-2, 1e-3, 1e-4])
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def test_scipy_not_install_error(tmp_path):
    scipy_module = keras_tuner.tuners.bayesian.scipy
    keras_tuner.tuners.bayesian.scipy = None

    with pytest.raises(ImportError, match="Please install scipy"):
        keras_tuner.BayesianOptimization(
            hypermodel=build_model,
            directory=tmp_path,
        )

    keras_tuner.tuners.bayesian.scipy = scipy_module


def test_sklearn_not_install_error(tmp_path):
    sklearn_module = keras_tuner.tuners.bayesian.sklearn
    keras_tuner.tuners.bayesian.sklearn = None

    with pytest.raises(ImportError, match="Please install scikit-learn"):
        keras_tuner.BayesianOptimization(
            hypermodel=build_model,
            directory=tmp_path,
        )

    keras_tuner.tuners.bayesian.sklearn = sklearn_module


def test_bayesian_oracle(tmp_path):
    hps = hp_module.HyperParameters()
    hps.Choice("a", [1, 2], default=1)
    hps.Int("b", 3, 10, default=3)
    hps.Float("c", 0, 1, 0.1, default=0)
    hps.Fixed("d", 7)
    hps.Choice("e", [9, 0], default=9)
    oracle = bo_module.BayesianOptimizationOracle(
        objective=keras_tuner.Objective("score", "max"),
        max_trials=20,
        num_initial_points=2,
        hyperparameters=hps,
    )
    oracle._set_project_dir(tmp_path, "untitled")
    for i in range(5):
        trial = oracle.create_trial(str(i))
        oracle.update_trial(trial.trial_id, {"score": i})
        trial.status = "COMPLETED"
        oracle.end_trial(trial)


def test_bayesian_oracle_with_zero_y(tmp_path):
    hps = hp_module.HyperParameters()
    hps.Choice("a", [1, 2], default=1)
    hps.Int("b", 3, 10, default=3)
    hps.Float("c", 0, 1, 0.1, default=0)
    hps.Fixed("d", 7)
    hps.Choice("e", [9, 0], default=9)
    oracle = bo_module.BayesianOptimizationOracle(
        objective=keras_tuner.Objective("score", "max"),
        max_trials=20,
        num_initial_points=2,
        hyperparameters=hps,
    )
    oracle._set_project_dir(tmp_path, "untitled")
    for i in range(5):
        trial = oracle.create_trial(str(i))
        oracle.update_trial(trial.trial_id, {"score": 0})
        trial.status = "COMPLETED"
        oracle.end_trial(trial)


def test_bayesian_dynamic_space(tmp_path):
    hps = hp_module.HyperParameters()
    hps.Choice("a", [1, 2], default=1)
    oracle = bo_module.BayesianOptimizationOracle(
        objective="val_acc", max_trials=20, num_initial_points=10
    )
    oracle._set_project_dir(tmp_path, "untitled")
    oracle.hyperparameters = hps
    for i in range(10):
        oracle.populate_space(str(i))
    hps.Int("b", 3, 10, default=3)
    assert "b" in oracle.populate_space("1_0")["values"]
    hps.Float("c", 0, 1, 0.1, default=0)
    assert "c" in oracle.populate_space("1_1")["values"]
    hps.Fixed("d", 7)
    assert "d" in oracle.populate_space("1_2")["values"]
    hps.Choice("e", [9, 0], default=9)
    assert "e" in oracle.populate_space("1_3")["values"]


def test_bayesian_save_reload(tmp_path):
    hps = hp_module.HyperParameters()
    hps.Choice("a", [1, 2], default=1)
    hps.Choice("b", [3, 4], default=3)
    hps.Choice("c", [5, 6], default=5)
    hps.Choice("d", [7, 8], default=7)
    hps.Choice("e", [9, 0], default=9)
    oracle = bo_module.BayesianOptimizationOracle(
        objective=keras_tuner.Objective("score", "max"),
        max_trials=20,
        hyperparameters=hps,
    )
    oracle._set_project_dir(tmp_path, "untitled")

    for _ in range(3):
        trial = oracle.create_trial("tuner_id")
        oracle.update_trial(trial.trial_id, {"score": 1.0})
        trial.status = "COMPLETED"
        oracle.end_trial(trial)

    oracle.save()
    oracle = bo_module.BayesianOptimizationOracle(
        objective=keras_tuner.Objective("score", "max"),
        max_trials=20,
        hyperparameters=hps,
    )
    oracle._set_project_dir(tmp_path, "untitled")
    oracle.reload()

    for _ in range(3):
        trial = oracle.create_trial("tuner_id")
        oracle.update_trial(trial.trial_id, {"score": 1.0})
        trial.status = "COMPLETED"
        oracle.end_trial(trial)

    assert len(oracle.trials) == 6


def test_bayesian_optimization_tuner(tmp_path):
    tuner = bo_module.BayesianOptimization(
        build_model, objective="val_accuracy", max_trials=15, directory=tmp_path
    )
    assert isinstance(tuner.oracle, bo_module.BayesianOptimizationOracle)


def test_bayesian_optimization_tuner_set_alpha_beta(tmp_path):
    tuner = bo_module.BayesianOptimization(
        build_model,
        alpha=1e-4,
        beta=2.6,
        objective="val_accuracy",
        max_trials=15,
        directory=tmp_path,
    )
    assert isinstance(tuner.oracle, bo_module.BayesianOptimizationOracle)


def test_save_before_result(tmp_path):
    hps = hp_module.HyperParameters()
    hps.Choice("a", [1, 2], default=1)
    hps.Int("b", 3, 10, default=3)
    hps.Float("c", 0, 1, 0.1, default=0)
    hps.Fixed("d", 7)
    hps.Choice("e", [9, 0], default=9)
    oracle = bo_module.BayesianOptimizationOracle(
        objective=keras_tuner.Objective("score", "max"),
        max_trials=10,
        hyperparameters=hps,
    )
    oracle._set_project_dir(tmp_path, "untitled")
    oracle.populate_space(str(1))
    oracle.save()


def test_bayesian_oracle_maximize(tmp_path):
    hps = hp_module.HyperParameters()
    hps.Int("a", -100, 100)

    oracle = bo_module.BayesianOptimizationOracle(
        objective=keras_tuner.Objective("score", direction="max"),
        max_trials=20,
        hyperparameters=hps,
        num_initial_points=2,
    )
    oracle._set_project_dir(tmp_path, "untitled")

    # Make examples with high 'a' and high score.
    for i in range(5):
        trial = trial_module.Trial(hyperparameters=hps.copy())
        trial.hyperparameters.values["a"] = 10 * i
        trial.score = i
        trial.status = "COMPLETED"
        oracle.trials[trial.trial_id] = trial

    # Make examples with low 'a' and low score
    for i in range(5):
        trial = trial_module.Trial(hyperparameters=hps.copy())
        trial.hyperparameters.values["a"] = -10 * i
        trial.score = -i
        trial.status = "COMPLETED"
        oracle.trials[trial.trial_id] = trial

    trial = oracle.create_trial("tuner0")
    assert trial.status == "RUNNING"
    # Assert that the oracle suggests hps it thinks will maximize.
    assert trial.hyperparameters.get("a") > 0


def test_hyperparameters_added(tmp_path):
    hps = hp_module.HyperParameters()
    hps.Int("a", -100, 100)

    oracle = bo_module.BayesianOptimizationOracle(
        objective=keras_tuner.Objective("score", direction="max"),
        max_trials=20,
        hyperparameters=hps,
        num_initial_points=2,
    )
    oracle._set_project_dir(tmp_path, "untitled")

    # Populate initial trials.
    for i in range(10):
        trial = trial_module.Trial(hyperparameters=hps.copy())
        trial.hyperparameters.values["a"] = 10 * i
        trial.score = i
        trial.status = "COMPLETED"
        oracle.trials[trial.trial_id] = trial

    # A new trial discovered a new hp and synced to oracle.hyperparameters.
    new_hps = hp_module.HyperParameters()
    new_hps.Float("b", 3.2, 6.4, step=0.2, default=3.6)
    new_hps.Boolean("c", default=True)
    oracle.update_space(new_hps)

    # Make a new trial, it should have b set.
    trial = oracle.create_trial("tuner0")
    assert trial.status == "RUNNING"
    assert "b" in trial.hyperparameters.values
    assert "c" in trial.hyperparameters.values


def test_step_respected(tmp_path):
    hps = hp_module.HyperParameters()
    hps.Float("c", 0, 10, step=3)
    oracle = bo_module.BayesianOptimizationOracle(
        objective=keras_tuner.Objective("score", direction="max"),
        max_trials=20,
        hyperparameters=hps,
        num_initial_points=2,
    )
    oracle._set_project_dir(tmp_path, "untitled")

    # Populate initial trials.
    for i in range(10):
        trial = trial_module.Trial(hyperparameters=hps.copy())
        trial.hyperparameters.values["c"] = 3.0
        trial.score = i
        trial.status = "COMPLETED"
        oracle.trials[trial.trial_id] = trial

    trial = oracle.create_trial("tuner0")
    # Check that oracle respects the `step` param.
    assert trial.hyperparameters.get("c") in {0, 3, 6, 9}


def test_float_optimization(tmp_path):
    class PolynomialTuner(keras_tuner.engine.base_tuner.BaseTuner):
        def run_trial(self, trial):
            hp = trial.hyperparameters
            return -1 * hp["a"] ** 3 + hp["b"] ** 3 + hp["c"] - abs(hp["d"])

    hps = hp_module.HyperParameters()
    hps.Float("a", -1, 1)
    hps.Float("b", -1, 1)
    hps.Float("c", -1, 1)
    hps.Float("d", -1, 1)

    tuner = PolynomialTuner(
        oracle=keras_tuner.oracles.BayesianOptimization(
            objective=keras_tuner.Objective("score", "max"),
            hyperparameters=hps,
            max_trials=50,
        ),
        directory=tmp_path,
    )

    tuner.search()

    atol, rtol = 1e-1, 1e-1
    best_trial = tuner.oracle.get_best_trials()[0]
    best_hps = best_trial.hyperparameters

    assert np.isclose(best_trial.score, 3, atol=atol, rtol=rtol)
    assert np.isclose(best_hps["a"], -1, atol=atol, rtol=rtol)
    assert np.isclose(best_hps["b"], 1, atol=atol, rtol=rtol)
    assert np.isclose(best_hps["c"], 1, atol=atol, rtol=rtol)
    assert np.isclose(best_hps["d"], 0, atol=atol, rtol=rtol)


def test_distributed_optimization(tmp_path):

    hps = hp_module.HyperParameters()
    hps.Int("a", 0, 10)
    hps.Float("b", -1, 1, step=0.1)
    hps.Float("c", 1e-5, 1e-2, sampling="log")

    def evaluate(hp):
        # Minimum at a=4, b=1, c=1e-3 with score=-1
        return abs(hp["a"] - 4) - hp["b"] + 0.1 * abs(3 + math.log(hp["c"], 10))

    oracle = bo_module.BayesianOptimizationOracle(
        objective=keras_tuner.Objective("score", "min"),
        hyperparameters=hps,
        max_trials=60,
    )
    oracle._set_project_dir(tmp_path, "untitled")

    tuners = 4

    for _ in range(10):
        trials = []
        for i in range(tuners):
            trial = oracle.create_trial(f"tuner_{str(i)}")
            trials.append(trial)
        for trial in trials:
            oracle.update_trial(
                trial.trial_id, {"score": evaluate(trial.hyperparameters)}
            )
        for trial in trials:
            trial.status = "COMPLETED"
            oracle.end_trial(trial)

    atol, rtol = 1e-1, 1e-1
    best_trial = oracle.get_best_trials()[0]
    best_hps = best_trial.hyperparameters

    # The minimum is not always found but it is always close.
    assert best_trial.score < -0.8, best_hps.values
    assert np.isclose(best_hps["a"], 4, atol=atol, rtol=rtol)
    assert np.isclose(best_hps["b"], 1, atol=atol, rtol=rtol)

    # For log-scale param, just check that the order of magnitude is correct.
    log_best_c = math.log(best_hps["c"], 10)
    assert log_best_c > -4 and log_best_c < -2


def test_interleaved_distributed_optimization(tmp_path):
    hps = hp_module.HyperParameters()
    hps.Float("a", -1, 1)
    hps.Float("b", -1, 1)
    hps.Float("c", -1, 1)
    hps.Float("d", -1, 1)

    def evaluate(hp):
        # Minimum at a=4, b=1, c=1e-3 with score=-1
        return -1 * hp["a"] ** 3 + hp["b"] ** 3 + hp["c"] - abs(hp["d"])

    oracle = bo_module.BayesianOptimizationOracle(
        objective=keras_tuner.Objective("score", "min"),
        hyperparameters=hps,
        max_trials=60,
        num_initial_points=2,
    )
    oracle._set_project_dir(tmp_path, "untitled")

    # Run 4 trials on 2 tuners

    # Start both tuners at the same time
    trial_1 = oracle.create_trial("tuner_0")
    trial_2 = oracle.create_trial("tuner_1")

    # tuner_0 finishes trial_1 before tuner_1 finishes
    oracle.update_trial(
        trial_1.trial_id, {"score": evaluate(trial_1.hyperparameters)}
    )
    trial_1.status = "COMPLETED"
    oracle.end_trial(trial_1)

    # tuner_0 request a new trial (trial_3)
    trial_3 = oracle.create_trial("tuner_0")

    # tuner_1 finishes trial_2
    oracle.update_trial(
        trial_2.trial_id, {"score": evaluate(trial_2.hyperparameters)}
    )
    trial_2.status = "COMPLETED"
    oracle.end_trial(trial_2)

    # tuner_1 requests the final new trial (trial_4)
    # the Bayesian optimizer will use ongoing trial_3 to hallucinate
    trial_4 = oracle.create_trial("tuner_1")

    # tuner_0 finishes trial_3
    oracle.update_trial(
        trial_3.trial_id, {"score": evaluate(trial_3.hyperparameters)}
    )
    trial_3.status = "COMPLETED"
    oracle.end_trial(trial_3)

    # tuner_1 finishes trial_4
    oracle.update_trial(
        trial_4.trial_id, {"score": evaluate(trial_4.hyperparameters)}
    )
    trial_4.status = "COMPLETED"
    oracle.end_trial(trial_4)

    assert True
