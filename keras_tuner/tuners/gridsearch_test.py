# Copyright 2022 The KerasTuner Authors
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

import random

from tensorflow import keras

from keras_tuner.engine import hypermodel
from keras_tuner.tuners import gridsearch


def test_that_exhaustive_space_is_explored(tmp_path):
    # Tests that it explores the whole search space given by the combination
    # of all hyperparameter of Choice type.

    # Given
    want_units_1 = [2, 4]
    want_units_2 = [1, 3]
    want_optimizers = ["adam", "sgd", "rmsprop"]
    want_loss = "binary_crossentropy"
    want_dropouts = [True, False]

    class MyGridSearch(gridsearch.GridSearch):
        def run_trial(self, trial, *args, **kwargs):
            hp = trial.hyperparameters
            hp.Choice("units_1", values=want_units_1)
            hp.Boolean("dropout", default=want_dropouts[0])
            hp.Choice("units_2", values=want_units_2)
            hp.Choice("optmizer", values=want_optimizers),
            hp.Fixed("loss", value=want_loss)
            return random.random()

    # When
    tuner = MyGridSearch(directory=tmp_path)
    tuner.search()

    # Then
    assert {hp.name for hp in tuner.oracle.get_space().space} == {
        "units_1",
        "optmizer",
        "units_2",
        "loss",
        "dropout",
    }

    # 2 units_1, 3 optimizers, 2 units_2, 2 dropout and 1 loss
    expected_hyperparameter_space = 24
    assert len(tuner.oracle.trials) == expected_hyperparameter_space

    trials = tuner.oracle.get_best_trials(num_trials=expected_hyperparameter_space)
    explored_space = [trial.hyperparameters.values for trial in trials]
    for want_unit_1 in want_units_1:
        for want_unit_2 in want_units_2:
            for want_optimizer in want_optimizers:
                for want_dropout in want_dropouts:
                    assert {
                        "units_1": want_unit_1,
                        "units_2": want_unit_2,
                        "optmizer": want_optimizer,
                        "loss": want_loss,
                        "dropout": want_dropout,
                    } in explored_space


def test_int_and_float(tmp_path):
    class MyGridSearch(gridsearch.GridSearch):
        def run_trial(self, trial, *args, **kwargs):
            hp = trial.hyperparameters
            hp.Int("int", 1, 5)
            hp.Float("float", 1, 2)
            return random.random()

    tuner = MyGridSearch(directory=tmp_path)
    tuner.search()
    # int has 5 values, float sampled 10 values and 1 default value
    # 5 * (10 + 1)
    assert len(tuner.oracle.trials) == 55


def test_new_hp(tmp_path):
    class MyGridSearch(gridsearch.GridSearch):
        def run_trial(self, trial, *args, **kwargs):
            hp = trial.hyperparameters
            if hp.Boolean("bool"):
                hp.Choice("choice1", [0, 1, 2])
            else:
                hp.Choice("choice2", [3, 4, 5])
            return random.random()

    tuner = MyGridSearch(directory=tmp_path)
    tuner.search(verbose=0)
    assert len(tuner.oracle.trials) == 3 + 3 * 3


def test_hp_in_fit(tmp_path):
    class MyHyperModel(hypermodel.HyperModel):
        def build(self, hp):
            hp.Fixed("fixed", 3)
            return keras.Sequential()

        def fit(self, hp, model, *args, **kwargs):
            hp.Choice("choice", [0, 1, 2])
            return random.random()

    tuner = gridsearch.GridSearch(hypermodel=MyHyperModel(), directory=tmp_path)
    tuner.search(verbose=0)
    assert len(tuner.oracle.trials) == 3


# def test_conditional_scope(tmp_path):
#     class MyHyperModel(hypermodel.HyperModel):
#         def build(self, hp):
#             a = hp.Boolean("bool")
#             with hp.conditional_scope("bool", [True]):
#                 if a:
#                     hp.Choice("choice1", [0, 1, 2])
#             with hp.conditional_scope("bool", [False]):
#                 if not a:
#                     hp.Choice("choice2", [3, 4, 5])
#             return keras.Sequential()

#         def fit(self, hp, model, *args, **kwargs):
#             a = hp.Boolean("bool2")
#             with hp.conditional_scope("bool2", [True]):
#                 if a:
#                     hp.Choice("choice3", [0, 1, 2])
#             with hp.conditional_scope("bool", [False]):
#                 if not a:
#                     hp.Choice("choice4", [3, 4, 5])
#             return random.random()

#     tuner = gridsearch.GridSearch(hypermodel=MyHyperModel(), directory=tmp_path)
#     tuner.search(verbose=0)
#     assert len(tuner.oracle.trials) == 36
