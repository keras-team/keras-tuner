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

import os

import numpy as np
import pytest
from tensorflow import keras

from keras_tuner.engine import objective as obj_module
from keras_tuner.engine import tuner_utils


def test_save_best_epoch_with_single_objective(tmp_path):
    objective = obj_module.create_objective("val_loss")
    filepath = os.path.join(tmp_path, "saved_weights")
    callback = tuner_utils.SaveBestEpoch(objective, filepath)

    model = keras.Sequential([keras.layers.Dense(1)])
    model.compile(loss="mse")
    val_x = np.random.rand(10, 10)
    val_y = np.random.rand(10, 10)
    history = model.fit(
        x=np.random.rand(10, 10),
        y=np.random.rand(10, 1),
        validation_data=(val_x, val_y),
        epochs=10,
        callbacks=[callback],
    )

    model.load_weights(filepath)

    assert min(history.history["val_loss"]) == model.evaluate(val_x, val_y)


def test_save_best_epoch_with_multi_objective(tmp_path):
    objective = obj_module.create_objective(["val_loss", "val_mae"])
    filepath = os.path.join(tmp_path, "saved_weights")
    callback = tuner_utils.SaveBestEpoch(objective, filepath)

    model = keras.Sequential([keras.layers.Dense(1)])
    model.compile(loss="mse", metrics=["mae"])
    val_x = np.random.rand(10, 10)
    val_y = np.random.rand(10, 10)
    history = model.fit(
        x=np.random.rand(10, 10),
        y=np.random.rand(10, 1),
        validation_data=(val_x, val_y),
        epochs=10,
        callbacks=[callback],
    )

    model.load_weights(filepath)

    assert min(history.history["val_loss"]) + min(history.history["val_mae"]) == sum(
        model.evaluate(val_x, val_y)
    )


def test_save_best_epoch_with_default_objective(tmp_path):
    objective = obj_module.create_objective(None)
    filepath = os.path.join(tmp_path, "saved_weights")
    callback = tuner_utils.SaveBestEpoch(objective, filepath)

    model = keras.Sequential([keras.layers.Dense(1)])
    model.compile(loss="mse")
    val_x = np.random.rand(10, 10)
    val_y = np.random.rand(10, 10)
    history = model.fit(
        x=np.random.rand(10, 10),
        y=np.random.rand(10, 1),
        validation_data=(val_x, val_y),
        epochs=10,
        callbacks=[callback],
    )

    model.load_weights(filepath)

    assert history.history["val_loss"][-1] == model.evaluate(val_x, val_y)


def test_convert_to_metrics_with_history():
    model = keras.Sequential([keras.layers.Dense(1)])
    model.compile(loss="mse", metrics=["mae"])
    val_x = np.random.rand(10, 10)
    val_y = np.random.rand(10, 10)
    history = model.fit(
        x=np.random.rand(10, 10),
        y=np.random.rand(10, 1),
        validation_data=(val_x, val_y),
    )

    results = tuner_utils.convert_to_metrics_dict(
        history,
        obj_module.Objective("val_loss", "min"),
    )
    assert all(key in results for key in ["loss", "val_loss", "mae", "val_mae"])


def test_convert_to_metrics_with_float():
    assert tuner_utils.convert_to_metrics_dict(
        0.1,
        obj_module.Objective("val_loss", "min"),
    ) == {"val_loss": 0.1}


def test_convert_to_metrics_with_dict():
    assert tuner_utils.convert_to_metrics_dict(
        {"loss": 0.2, "val_loss": 0.1},
        obj_module.Objective("val_loss", "min"),
    ) == {"loss": 0.2, "val_loss": 0.1}


def test_convert_to_metrics_with_list_of_floats():
    assert tuner_utils.convert_to_metrics_dict(
        [0.1, 0.2],
        obj_module.Objective("val_loss", "min"),
    ) == {"val_loss": (0.1 + 0.2) / 2}


def test_convert_to_metrics_with_dict_without_obj_key():
    with pytest.raises(ValueError, match="the specified objective"):
        tuner_utils.validate_trial_results(
            {"loss": 0.1}, obj_module.Objective("val_loss", "min"), "func_name"
        )


def test_get_best_step_return_zero():
    assert (
        tuner_utils.get_best_step(
            [{"val_loss": 1}, {"val_loss": 2}],
            obj_module.Objective("val_loss", "min"),
        )
        == 0
    )


def test_get_best_step_return_average_epoch():
    class History(keras.callbacks.History):
        def __init__(self, history):
            self.history = history

    results = [
        History(
            {
                "val_loss": [5, 8, 3, 1, 2],
                "val_accuracy": [5, 8, 3, 1, 2],
            }
        ),
        History(
            {
                "val_loss": [5, 8, 3, 2, 1],
                "val_accuracy": [5, 8, 3, 1, 2],
            }
        ),
    ]
    assert (
        tuner_utils.get_best_step(
            results,
            obj_module.Objective("val_loss", "min"),
        )
        == 3
    )
