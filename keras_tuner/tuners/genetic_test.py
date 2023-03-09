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
import tensorflow as tf

import keras_tuner
from keras_tuner.engine import hyperparameters as hp_module
from keras_tuner.tuners import genetic as ge_module


def build_model(hp):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=(2, 2)))
    for i in range(3):
        model.add(
            tf.keras.layers.Dense(
                units=hp.Int("units_" + str(i), 2, 4, 2), activation="relu"
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


def test_mutation():
    """Test mutation of a chromosome."""
    hps = hp_module.HyperParameters()
    hps.Choice("a", [1, 2, 3])
    hps.Float("b", 0, 1)
    hps.Int("c", 0, 10)
    hps.Fixed("d", 1)

    gep = ge_module.GeneticEvolutionaryProcess(
        mutation_factor=1.1, crossover_factor=0.5
    )
    mutated = gep._mutate(hps)
    assert mutated.values != hps.values
    assert mutated.get("a") in [1, 2, 3]
    assert 0 <= mutated.get("b") <= 1
    assert 0 <= mutated.get("c") <= 10
    assert mutated.get("d") == 1


def test_no_mutation():
    """Test that no mutation occurs when mutation factor is 0."""
    hps = hp_module.HyperParameters()
    hps.Choice("a", [1, 2, 3])
    hps.Float("b", 0, 1)
    hps.Int("c", 0, 10)
    hps.Fixed("d", 1)

    gep = ge_module.GeneticEvolutionaryProcess(
        mutation_factor=-1, crossover_factor=0.5
    )
    mutated = gep._mutate(hps)
    assert mutated.values == hps.values


def test_crossover():
    """Test crossover of two chromosomes."""
    hp1 = hp_module.HyperParameters()
    hp1.Choice("a", [1, 2, 3])
    hp1.Float("b", 0, 1, step=0.1)
    hp1.Int("c", 0, 10, step=2)
    hp1.Fixed("d", 1)

    hp2 = hp_module.HyperParameters()
    hp2.Choice("a", [4, 5, 1])
    hp2.Float("b", -1, 0, step=0.1)
    hp2.Int("c", 10, 20, step=2)
    hp2.Fixed("d", 2)

    gep = ge_module.GeneticEvolutionaryProcess(
        mutation_factor=0, crossover_factor=1.1
    )
    parent_1, parent_2 = gep._crossover(hp1, hp2)
    assert parent_1.values != hp1.values
    assert parent_2.values != hp2.values
    assert 0 <= parent_1.get("b") <= 1
    assert -1 <= parent_2.get("b") <= 0


def test_no_crossover():
    """Test that no crossover occurs when crossover factor is 0."""
    hp1 = hp_module.HyperParameters()
    hp1.Choice("a", [1, 2, 3])
    hp1.Float("b", 0, 1)
    hp1.Int("c", 0, 10)
    hp1.Fixed("d", 1)

    hp2 = hp_module.HyperParameters()
    hp2.Choice("a", [4, 5, 1])
    hp2.Float("b", -1, 0)
    hp2.Int("c", 10, 20)
    hp2.Fixed("d", 2)

    gep = ge_module.GeneticEvolutionaryProcess(
        mutation_factor=0, crossover_factor=-1
    )
    parent_1, parent_2 = gep._crossover(hp1, hp2)
    assert parent_1.values == hp1.values
    assert parent_2.values == hp2.values


def test_make_ranges_without_offspring_size():
    """Test that the ranges are created correctly."""

    goo = ge_module.GeneticOptimizationOracle(
        objective=keras_tuner.Objective("val_accuracy", "max"),
        generation_size=2,
        population_size=10,
        offspring_size=None,
    )

    goo._make_ranges
    assert goo.max_trials == 10 + 2 * 10 * 2
    assert goo.population_range == list(range(10))
    assert goo.generation_range == list(range(30, 50, 19))
    assert goo.first_parent_range == list(range(10, 50))[::2]
    assert goo.second_parent_range == list(range(10, 50))[1::2]


def test_make_with_offspring_size():
    """Test that the ranges are created correctly."""

    goo = ge_module.GeneticOptimizationOracle(
        objective=keras_tuner.Objective("val_accuracy", "max"),
        generation_size=2,
        population_size=10,
        offspring_size=5,
    )

    goo._make_ranges
    assert goo.max_trials == 10 + 2 * 5 * 2
    assert goo.population_range == list(range(10))
    assert goo.generation_range == list(range(20, 30, 9))
    assert goo.first_parent_range == list(range(10, 30))[::2]
    assert goo.second_parent_range == list(range(10, 30))[1::2]


def test_raises_factor():
    """Test that the crossover and mutation factors are in the correct limit."""

    with pytest.raises(ValueError) as excinfo:
        ge_module.GeneticOptimizationOracle(
            objective=keras_tuner.Objective("val_accuracy", "max"),
            generation_size=2,
            population_size=21,
            offspring_size=5,
            mutation_factor=1.2,
            crossover_factor=0.5,
        )

    assert (
        "The sum of the 'mutation_factors' and "
        "'crossover_factors' must be less than 1.0." in str(excinfo.value)
    )


def test_raises_selection():
    """Test that the selection method is correct."""

    with pytest.raises(ValueError) as excinfo:
        ge_module.GeneticOptimizationOracle(
            objective=keras_tuner.Objective("val_accuracy", "max"),
            generation_size=2,
            population_size=21,
            offspring_size=5,
            selection_type="random",
        )
    assert (
        "The 'selection_type' must be either 'roulette_wheel' or "
        "'tournament'." in str(excinfo.value)
    )


def test_roulette_wheel_selection():
    """Test that the roulette wheel selection works correctly."""

    scores = [1.0, 2.0, 3.0, 4.0, 5.0]
    hps = hp_module.HyperParameters()
    hps.Choice("a", [1, 2, 3])
    hps.Float("b", 0, 1)
    hps.Int("c", 0, 10)
    hps.Fixed("d", 1)

    gep = ge_module.GeneticEvolutionaryProcess(
        mutation_factor=1, crossover_factor=0
    )
    population = [gep._mutate(hps) for _ in range(5)]

    parent_1, parent_2 = gep._roulette_wheel_selection(
        scores=scores, population=population
    )
    assert parent_1.values != parent_2.values
    assert len(parent_1.values) == len(parent_2.values)
    assert len(parent_1.values) == len(hps.values)
    assert len(parent_2.values) == len(hps.values)
    assert parent_1.values in [hp.values for hp in population]
    assert parent_2.values in [hp.values for hp in population]


def test_tournament_selection():
    """Test that the tournament selection works correctly."""

    scores = [1.0, 2.0, 3.0, 4.0, 5.0]
    hps = hp_module.HyperParameters()
    hps.Choice("a", [1, 2, 3])
    hps.Float("b", 0, 1)
    hps.Int("c", 0, 10)
    hps.Fixed("d", 1)

    gep = ge_module.GeneticEvolutionaryProcess(
        mutation_factor=1, crossover_factor=0
    )
    population = [gep._mutate(hps) for _ in range(5)]

    parent_1, parent_2 = gep._tournament_selection(
        scores=scores, population=population
    )
    assert parent_1.values != parent_2.values
    assert len(parent_1.values) == len(parent_2.values)
    assert len(parent_1.values) == len(hps.values)
    assert len(parent_2.values) == len(hps.values)
    assert parent_1.values in [hp.values for hp in population]
    assert parent_2.values in [hp.values for hp in population]


def test_populate_space_init(tmp_path):
    """Test that the population is initialized correctly."""

    hps = hp_module.HyperParameters()
    hps.Choice("a", [1, 2, 3])
    hps.Float("b", 0, 1)
    hps.Int("c", 0, 10)
    hps.Fixed("d", 1)

    oracle = ge_module.GeneticOptimizationOracle(
        objective=keras_tuner.Objective("val_accuracy", "max"),
        generation_size=5,
        population_size=10,
        offspring_size=5,
        hyperparameters=hps,
    )
    oracle._set_project_dir(tmp_path, "untitled")
    oracle._populate_space("00")
    assert oracle.population_size == 10
    assert oracle.offspring_size == 5


def test_populate_inits(tmp_path):
    """Test that the initiale batch is true random."""

    hps = hp_module.HyperParameters()
    hps.Choice("a", [1, 2, 3])
    hps.Float("b", 0, 1)
    hps.Int("c", 0, 10)
    hps.Fixed("d", 1)

    oracle = ge_module.GeneticOptimizationOracle(
        objective=keras_tuner.Objective("val_accuracy", "max"),
        generation_size=5,
        population_size=10,
        offspring_size=5,
        hyperparameters=hps,
    )
    oracle._set_project_dir(tmp_path, "untitled")
    for _ in range(10):
        trial = oracle.create_trial("trial_id")
        oracle.update_trial(trial.trial_id, {"val_accuracy": 0.2})
        oracle.end_trial(trial.trial_id, "COMPLETED")

    assert (
        oracle.population["hyperparameters"][1]
        != oracle.population["hyperparameters"][2]
    )
    assert (
        oracle.population["hyperparameters"][1]
        != oracle.population["hyperparameters"][3]
    )
    assert oracle.population["scores"][1] == oracle.population["scores"][4]


def test_score_early_ridged_min(tmp_path):
    """Test early stopping with a min objective."""

    hps = hp_module.HyperParameters()
    hps.Choice("a", [1, 2, 3])
    hps.Float("b", 0, 1)
    hps.Int("c", 0, 10)
    hps.Fixed("d", 1)

    oracle = ge_module.GeneticOptimizationOracle(
        objective=keras_tuner.Objective("val_accuracy", "min"),
        generation_size=5,
        population_size=10,
        offspring_size=5,
        hyperparameters=hps,
        threshold=0.1,
    )
    oracle._set_project_dir(tmp_path, "untitled")
    for _ in range(2):
        trial = oracle.create_trial("trial_id")
        oracle.update_trial(trial.trial_id, {"val_accuracy": 0.2})
        oracle.end_trial(trial.trial_id, "COMPLETED")

    _trial = oracle._check_score(0.01)
    assert _trial["status"] == "COMPLETED"
    assert _trial["values"] != hps.values


def test_score_stopped(tmp_path):
    """Test early stopping with a min objective."""

    hps = hp_module.HyperParameters()
    hps.Choice("a", [1, 2, 3])
    hps.Float("b", 0, 1)
    hps.Int("c", 0, 10)
    hps.Fixed("d", 1)

    oracle = ge_module.GeneticOptimizationOracle(
        objective=keras_tuner.Objective("val_accuracy", "min"),
        generation_size=5,
        population_size=10,
        offspring_size=5,
        hyperparameters=hps,
        threshold=0.1,
    )
    oracle._set_project_dir(tmp_path, "untitled")
    for _ in range(1):
        trial = oracle.create_trial("trial_id")
        oracle.update_trial(trial.trial_id, {"val_accuracy": 0.2})
        oracle.end_trial(trial.trial_id, "COMPLETED")

    oracle.values = None
    oracle.start_order = []
    _trial = oracle.populate_space("00")
    assert _trial["status"] == "STOPPED"
    assert _trial["values"] is None


def test_get_set_state(tmp_path):
    """Test that the state is correctly set and retrieved."""

    hps = hp_module.HyperParameters()
    hps.Choice("a", [1, 2, 3])
    hps.Float("b", 0, 1)
    hps.Int("c", 0, 10)
    hps.Fixed("d", 1)

    oracle = ge_module.GeneticOptimizationOracle(
        objective=keras_tuner.Objective("val_accuracy", "min"),
        generation_size=5,
        population_size=10,
        offspring_size=5,
        hyperparameters=hps,
        threshold=0.1,
    )
    oracle._set_project_dir(tmp_path, "untitled")
    for _ in range(4):
        trial = oracle.create_trial("trial_id")
        oracle.update_trial(trial.trial_id, {"val_accuracy": 0.2})
        oracle.end_trial(trial.trial_id, "COMPLETED")

    state = oracle.get_state()
    oracle.set_state(state)
    assert oracle.population_size == 10
    assert oracle.offspring_size == 5
    assert oracle.generation_size == 5
    assert oracle.threshold == 0.1
    assert oracle.population["scores"][0] == 0.2
    assert oracle.population["scores"][1] == 0.2
    assert oracle.population["hyperparameters"][0] != hps.values
    assert oracle.population["hyperparameters"][1] != hps.values
    assert (
        oracle.population["hyperparameters"][0]
        != oracle.population["hyperparameters"][1]
    )
    assert (
        oracle.population["hyperparameters"][0]
        != oracle.population["hyperparameters"][2]
    )
    assert (
        oracle.population["hyperparameters"][1]
        != oracle.population["hyperparameters"][2]
    )


def test_tournament(tmp_path):
    """Test that the tournament selection works correctly."""

    hps = hp_module.HyperParameters()
    hps.Choice("a", [1, 2, 3])
    hps.Float("b", 0, 1, step=0.1)
    hps.Int("c", 0, 10, step=9)
    hps.Fixed("d", 1)

    oracle = ge_module.GeneticOptimizationOracle(
        objective=keras_tuner.Objective("val_accuracy", "max"),
        generation_size=5,
        population_size=10,
        offspring_size=5,
        hyperparameters=hps,
        selection_type="tournament",
    )
    oracle._set_project_dir(tmp_path, "untitled")
    for _ in range(60):
        trial = oracle.create_trial("trial_id")
        oracle.update_trial(trial.trial_id, {"val_accuracy": 0.2})
        oracle.end_trial(trial.trial_id, "COMPLETED")

    assert (
        oracle.population["hyperparameters"][1]
        != oracle.population["hyperparameters"][2]
    )
    assert (
        oracle.population["hyperparameters"][1]
        != oracle.population["hyperparameters"][3]
    )
    assert oracle.population["scores"][1] == oracle.population["scores"][4]


def test_roulette(tmp_path):
    """Test that the roulette selection works correctly."""

    hps = hp_module.HyperParameters()
    hps.Choice("a", [1, 2, 3])
    hps.Float("b", 0, 1, step=0.1)
    hps.Int("c", 0, 10, step=9)
    hps.Fixed("d", 1)

    oracle = ge_module.GeneticOptimizationOracle(
        objective=keras_tuner.Objective("val_accuracy", "max"),
        generation_size=5,
        population_size=10,
        offspring_size=5,
        hyperparameters=hps,
        selection_type="roulette_wheel",
    )
    oracle._set_project_dir(tmp_path, "untitled")
    for _ in range(60):
        trial = oracle.create_trial("trial_id")
        oracle.update_trial(trial.trial_id, {"val_accuracy": 0.2})
        oracle.end_trial(trial.trial_id, "COMPLETED")

    assert (
        oracle.population["hyperparameters"][1]
        != oracle.population["hyperparameters"][2]
    )
    assert (
        oracle.population["hyperparameters"][1]
        != oracle.population["hyperparameters"][3]
    )
    assert oracle.population["scores"][1] == oracle.population["scores"][4]


def test_genetic_minimize_tournament(tmp_path):
    """Test that the tournament selection works correctly for minimization."""

    class MyTuner(keras_tuner.GeneticOptimization):
        def run_trial(self, trial, *args, **kwargs):
            # Get the hp from trial.
            hp = trial.hyperparameters
            # Define "x" as a hyperparameter.
            x = hp.Float("x", min_value=-1.0, max_value=1.0, step=0.001)
            # Return the objective value to minimize.
            return x * x + 1

    tuner = MyTuner(
        # No hypermodel or objective specified.
        overwrite=True,
        directory=tmp_path,
        project_name="tune_anything",
        objective=keras_tuner.Objective("score", "min"),
        population_size=10,
        offspring_size=5,
        generation_size=5,
        mutation_factor=0.9,
        crossover_factor=0.1,
        selection_type="tournament",
    )

    # No need to pass anything to search()
    # unless you use them in run_trial().
    tuner.search()
    assert tuner.oracle.selection_type == "tournament"
    assert np.isclose(
        tuner.get_best_hyperparameters()[0].get("x"), 0.0, atol=1e-1
    )


def test_genetic_minimize_roulette(tmp_path):
    """Test that the roulette selection works correctly for minimization."""

    class MyTuner(keras_tuner.GeneticOptimization):
        def run_trial(self, trial, *args, **kwargs):
            # Get the hp from trial.
            hp = trial.hyperparameters
            # Define "x" as a hyperparameter.
            x = hp.Float("x", min_value=-1.0, max_value=1.0, step=0.001)
            # Return the objective value to minimize.
            return x * x + 1

    tuner = MyTuner(
        # No hypermodel or objective specified.
        overwrite=True,
        directory=tmp_path,
        project_name="tune_anything",
        objective=keras_tuner.Objective("score", "min"),
        population_size=10,
        offspring_size=5,
        generation_size=5,
        mutation_factor=0.9,
        crossover_factor=0.1,
        selection_type="roulette_wheel",
    )

    # No need to pass anything to search()
    # unless you use them in run_trial().
    tuner.search()
    assert tuner.oracle.selection_type == "roulette_wheel"
    assert np.isclose(
        tuner.get_best_hyperparameters()[0].get("x"), 0.0, atol=1e-1
    )


def test_genetic_maximize_tournament(tmp_path):
    """Test that the tournament selection works correctly for maximization."""

    class MyTuner(keras_tuner.GeneticOptimization):
        def run_trial(self, trial, *args, **kwargs):
            # Get the hp from trial.
            hp = trial.hyperparameters
            # Define "x" as a hyperparameter.
            x = hp.Float("x", min_value=-1.0, max_value=1.0, step=0.001)
            # Return the objective value to maximize.
            return x * x + 1

    tuner = MyTuner(
        # No hypermodel or objective specified.
        overwrite=True,
        directory=tmp_path,
        project_name="tune_anything",
        objective=keras_tuner.Objective("score", "max"),
        population_size=10,
        offspring_size=5,
        generation_size=5,
        mutation_factor=0.9,
        crossover_factor=0.1,
        selection_type="tournament",
    )

    # No need to pass anything to search()
    # unless you use them in run_trial().
    tuner.search()
    assert tuner.oracle.selection_type == "tournament"
    assert np.isclose(
        tuner.get_best_hyperparameters()[0].get("x"), -1.0, atol=1e-1
    )


def test_genetic_maximize_roulette(tmp_path):
    """Test that the roulette selection works correctly for maximization."""

    class MyTuner(keras_tuner.GeneticOptimization):
        def run_trial(self, trial, *args, **kwargs):
            # Get the hp from trial.
            hp = trial.hyperparameters
            # Define "x" as a hyperparameter.
            x = hp.Float("x", min_value=-1.0, max_value=1.0, step=0.001)
            # Return the objective value to maximize.
            return x * x + 1

    tuner = MyTuner(
        # No hypermodel or objective specified.
        overwrite=True,
        directory=tmp_path,
        project_name="tune_anything",
        objective=keras_tuner.Objective("score", "max"),
        population_size=10,
        offspring_size=5,
        generation_size=5,
        mutation_factor=0.9,
        crossover_factor=0.1,
        selection_type="roulette_wheel",
    )

    # No need to pass anything to search()
    # unless you use them in run_trial().
    tuner.search()
    assert tuner.oracle.selection_type == "roulette_wheel"
    assert np.isclose(
        tuner.get_best_hyperparameters()[0].get("x"), -1.0, atol=1e-1
    )


def test_genetic_minimize_tournament_with_hypermodel(tmp_path):
    """Test that the tournament selection works correctly for cubic parable."""

    class DropWaveTuner(keras_tuner.GeneticOptimization):
        def run_trial(self, trial, *args, **kwargs):
            # Get the hp from trial.
            hp = trial.hyperparameters
            # Define "x" as a hyperparameter.
            x = hp.Float("x", min_value=-2.2, max_value=2.2, step=0.01)
            y = hp.Float("y", min_value=-2.2, max_value=2.2, step=0.01)
            # Return the objective value to minimize.
            return self.drop_wave(x, y)

        def drop_wave(self, x, y):
            # with global minima at -1, 2
            return (x + 1) ** 2 + (y - 2) ** 2 + np.sin(x * y)

    tuner = DropWaveTuner(
        overwrite=True,
        directory=tmp_path,
        project_name="tune_anything",
        objective=keras_tuner.Objective("score", "min"),
        population_size=20,
        offspring_size=25,
        generation_size=50,
        mutation_factor=0.8,
        crossover_factor=0.2,
        selection_type="roulette_wheel",
        threshold=0.1,
    )
    tuner.search()
    assert tuner.oracle.selection_type == "roulette_wheel"
    assert np.isclose(
        tuner.get_best_hyperparameters()[0].get("x"), -1.0, atol=1e-1
    )
    assert np.isclose(
        tuner.get_best_hyperparameters()[0].get("y"), 2.0, atol=1e-1
    )


def test_genetic_minimize_roulette_with_hypermodel(tmp_path):
    """Test the roulette selection works correctly for a drop wave."""

    class TrigTuner(keras_tuner.GeneticOptimization):
        def run_trial(self, trial, *args, **kwargs):
            # Get the hp from trial.
            hp = trial.hyperparameters
            # Define "x" as a hyperparameter.
            x = hp.Float("x", min_value=-0.5, max_value=0.5, step=0.01)
            y = hp.Float("y", min_value=-0.5, max_value=0.5, step=0.01)
            # Return the objective value to minimize.
            return self.drop_wave(x, y)

        def drop_wave(self, x, y):
            # With global minima at 0 and 0
            return -(
                1
                + np.cos(12 * np.sqrt(x**2 + y**2))
                / (0.5 * (x**2 + y**2) + 2)
            )

    tuner = TrigTuner(
        overwrite=True,
        directory=tmp_path,
        project_name="tune_anything",
        objective=keras_tuner.Objective("score", "min"),
        population_size=20,
        offspring_size=35,
        generation_size=50,
        mutation_factor=0.6,
        crossover_factor=0.4,
        selection_type="roulette_wheel",
        threshold=0.05,
    )
    tuner.search()
    assert tuner.oracle.selection_type == "roulette_wheel"
    assert np.isclose(
        tuner.get_best_hyperparameters()[0].get("x"), 0.0, atol=1e-1
    )
    assert np.isclose(
        tuner.get_best_hyperparameters()[0].get("y"), 0.0, atol=1e-1
    )
