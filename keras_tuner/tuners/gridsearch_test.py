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

import numpy as np
import tensorflow as tf

from keras_tuner.engine import tuner as tuner_module
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

    def build_model(hp):
        model = tf.keras.Sequential()
        model.add(
            tf.keras.layers.Dense(
                units=hp.Choice("units_1", values=want_units_1), activation="relu"
            )
        )
        if hp.Boolean("dropout", default=want_dropouts[0]):
            model.add(tf.keras.layers.Dropout(rate=0.25))
        model.add(
            tf.keras.layers.Dense(
                units=hp.Choice("units_2", values=want_units_2), activation="relu"
            )
        )
        model.add(tf.keras.layers.Dense(1, activation="sigmoid"))
        model.compile(
            hp.Choice("optmizer", values=want_optimizers),
            loss=hp.Fixed("loss", value=want_loss),
            metrics=["accuracy"],
        )
        return model

    class MyGridSearch(gridsearch.GridSearchOracle):
        populate_space_call_count = 0

        def populate_space(self, trial_id):
            result = super(MyGridSearch, self).populate_space(trial_id)
            MyGridSearch.populate_space_call_count += 1
            return result

    # When
    tuner = tuner_module.Tuner(
        oracle=MyGridSearch(objective="accuracy"),
        hypermodel=build_model,
        directory=tmp_path,
    )

    x, y = np.ones((10, 10)), np.ones((10, 1))
    tuner.search(x, y, epochs=1)

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
    assert tuner.oracle.populate_space_call_count == expected_hyperparameter_space

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
