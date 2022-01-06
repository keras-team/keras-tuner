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
import tensorflow as tf

from keras_tuner.engine import tuner as tuner_module
from keras_tuner.tuners import randomsearch


def test_update_space(tmp_path):
    # Tests that HyperParameters added after the first call to `build_model`
    # are sent to the Oracle via oracle.update_space.
    def build_model(hp):
        model = tf.keras.Sequential()
        for i in range(hp.Int("layers", 0, 2)):
            model.add(
                tf.keras.layers.Dense(
                    units=hp.Int("units_" + str(i), 2, 4, 2), activation="relu"
                )
            )
        model.add(tf.keras.layers.Dense(1, activation="sigmoid"))
        model.compile("adam", loss="binary_crossentropy", metrics=["accuracy"])
        return model

    class MyRandomSearch(randomsearch.RandomSearchOracle):
        def populate_space(self, trial_id):
            result = super(MyRandomSearch, self).populate_space(trial_id)
            if "values" in result:
                result["values"]["layers"] = 2
            return result

    tuner = tuner_module.Tuner(
        oracle=MyRandomSearch(objective="accuracy", max_trials=1),
        hypermodel=build_model,
        directory=tmp_path,
    )

    assert {hp.name for hp in tuner.oracle.get_space().space} == {"layers"}

    x, y = np.ones((10, 10)), np.ones((10, 1))
    tuner.search(x, y, epochs=1)

    assert {hp.name for hp in tuner.oracle.get_space().space} == {
        "layers",
        "units_0",
        "units_1",
    }
