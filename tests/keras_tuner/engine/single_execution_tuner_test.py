# Copyright 2019 The Keras Tuner Authors
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

import unittest

import numpy as np
import pytest
import tensorflow as tf
from tensorflow import keras

import keras_tuner
from keras_tuner.engine import single_execution_tuner

INPUT_DIM = 2
NUM_CLASSES = 3
NUM_SAMPLES = 64
TRAIN_INPUTS = np.random.random(size=(NUM_SAMPLES, INPUT_DIM))
TRAIN_TARGETS = np.random.randint(0, NUM_CLASSES, size=(NUM_SAMPLES, 1))
VAL_INPUTS = np.random.random(size=(NUM_SAMPLES, INPUT_DIM))
VAL_TARGETS = np.random.randint(0, NUM_CLASSES, size=(NUM_SAMPLES, 1))


@pytest.fixture(scope="function")
def tmp_dir(tmpdir_factory):
    return tmpdir_factory.mktemp("integration_test", numbered=True)


def build_model(hp):
    model = keras.Sequential(
        [
            keras.layers.Dense(
                hp.Int("units", 100, 1000, 100),
                input_shape=(INPUT_DIM,),
                activation="relu",
            ),
            keras.layers.Dense(NUM_CLASSES, activation="softmax"),
        ]
    )
    model.compile("rmsprop", "sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def test_update_trial(tmp_dir):
    class MyOracle(keras_tuner.Oracle):
        def populate_space(self, _):
            values = {p.name: p.random_sample() for p in self.hyperparameters.space}
            return {"values": values, "status": "RUNNING"}

        def update_trial(self, trial_id, metrics, step=0):
            if step == 3:
                trial = self.trials[trial_id]
                trial.status = "STOPPED"
                return trial.status
            return super(MyOracle, self).update_trial(trial_id, metrics, step)

    my_oracle = MyOracle(objective="val_accuracy", max_trials=2)
    tuner = single_execution_tuner.SingleExecutionTuner(
        oracle=my_oracle, hypermodel=build_model, directory=tmp_dir
    )
    tuner.search(
        x=TRAIN_INPUTS,
        y=TRAIN_TARGETS,
        epochs=5,
        validation_data=(VAL_INPUTS, VAL_TARGETS),
    )

    assert len(my_oracle.trials) == 2

    for trial in my_oracle.trials.values():
        # Test that early stopping worked.
        assert len(trial.metrics.get_history("val_accuracy")) == 3


def test_checkpoint_removal(tmp_dir):
    def build_model(hp):
        model = keras.Sequential(
            [keras.layers.Dense(hp.Int("size", 5, 10)), keras.layers.Dense(1)]
        )
        model.compile("sgd", "mse", metrics=["accuracy"])
        return model

    tuner = single_execution_tuner.SingleExecutionTuner(
        oracle=keras_tuner.tuners.randomsearch.RandomSearchOracle(
            objective="val_accuracy", max_trials=1, seed=1337
        ),
        hypermodel=build_model,
        directory=tmp_dir,
    )
    x, y = np.ones((1, 5)), np.ones((1, 1))
    tuner.search(x, y, validation_data=(x, y), epochs=21)
    trial = list(tuner.oracle.trials.values())[0]
    trial_id = trial.trial_id
    assert tf.io.gfile.exists(tuner._get_checkpoint_fname(trial_id, 20))
    assert not tf.io.gfile.exists(tuner._get_checkpoint_fname(trial_id, 10))


def save_model_setup_tuner(tmp_dir):
    class MyTuner(single_execution_tuner.SingleExecutionTuner):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.was_called = False

        def _delete_checkpoint(self, trial_id, epoch):
            self.was_called = True

        def _checkpoint_model(self, model, trial_id, epoch):
            pass

    class MyOracle(keras_tuner.engine.oracle.Oracle):
        def get_trial(self, trial_id):
            trial = unittest.mock.Mock()
            trial.metrics = unittest.mock.Mock()
            trial.metrics.get_best_step.return_value = 5
            return trial

    return MyTuner(
        oracle=MyOracle(objective="val_accuracy"),
        hypermodel=build_model,
        directory=tmp_dir,
    )


def test_save_model_delete_not_called(tmp_dir):
    tuner = save_model_setup_tuner(tmp_dir)
    tuner.save_model("a", None, step=15)
    assert not tuner.was_called


def test_save_model_delete_called(tmp_dir):
    tuner = save_model_setup_tuner(tmp_dir)
    tuner.save_model("a", None, step=16)
    assert tuner.was_called
