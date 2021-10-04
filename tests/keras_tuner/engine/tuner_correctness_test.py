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
from collections import namedtuple

import numpy as np
import pytest
import tensorflow as tf
from packaging.version import parse
from tensorflow import keras

import keras_tuner
from keras_tuner.engine import tuner as tuner_module

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


class MockModel(keras.Model):
    def __init__(self, full_history):
        super(MockModel, self).__init__()
        self.full_history = full_history
        self.callbacks = []
        self.optimizer = True

    def call_callbacks(self, callbacks, method_name, *args, **kwargs):
        for callback in callbacks:
            method = getattr(callback, method_name)
            method(*args, **kwargs)

    def on_epoch_begin(self, epoch):
        for callback in self.callbacks:
            callback.on_epoch_begin(epoch, logs=None)

    def on_epoch_end(self, epoch):
        logs = {"loss": np.average(self.full_history[epoch])}
        for callback in self.callbacks:
            callback.on_epoch_end(epoch, logs=logs)

    def on_batch_begin(self, epoch, batch):
        for callback in self.callbacks:
            callback.on_batch_begin(batch, logs=None)

    def on_batch_end(self, epoch, batch):
        logs = {"loss": self.full_history[epoch][batch]}
        for callback in self.callbacks:
            callback.on_batch_end(epoch, logs=logs)

    def fit(self, *args, **kwargs):
        self.callbacks = kwargs["callbacks"]
        for callback in self.callbacks:
            callback.model = self
        for epoch in range(len(self.full_history)):
            self.on_epoch_begin(epoch)
            for batch in range(len(self.full_history[epoch])):
                self.on_batch_begin(epoch, batch)
                self.on_batch_end(epoch, batch)
            self.on_epoch_end(epoch)
        History = namedtuple("History", "history")
        return History(
            {
                "loss": [
                    np.average(epoch_values) for epoch_values in self.full_history
                ]
            }
        )

    def save_weights(self, fname, **kwargs):
        pass

    def get_config(self):
        return {}


class MockHyperModel(keras_tuner.HyperModel):

    mode_0 = [[10, 9, 8], [7, 6, 5], [4, 3, 2]]
    mode_1 = [[13, 13, 13], [12, 12, 12], [11, 11, 11]]

    def __init__(self):
        # The first call to `build` in tuner __init__
        # will reset this to 0
        self.mode_0_execution_count = -1

    def build(self, hp):
        if hp.Choice("mode", [0, 1]) == 0:
            return MockModel(self.mode_0)
        return MockModel(self.mode_1)


def test_tuning_correctness(tmp_dir):
    tuner = keras_tuner.Tuner(
        oracle=keras_tuner.tuners.randomsearch.RandomSearchOracle(
            objective="loss", max_trials=2, seed=1337
        ),
        hypermodel=MockHyperModel(),
        directory=tmp_dir,
    )
    tuner.search()
    assert len(tuner.oracle.trials) == 2

    m0_epochs = [float(np.average(x)) for x in MockHyperModel.mode_0]
    m1_epochs = [float(np.average(x)) for x in MockHyperModel.mode_1]

    # Score tracking correctness
    first_trial, second_trial = sorted(
        tuner.oracle.trials.values(), key=lambda t: t.score
    )
    assert first_trial.score == min(m0_epochs)
    assert second_trial.score == min(m1_epochs)
    assert tuner.oracle.get_best_trials(1)[0].trial_id == first_trial.trial_id


def test_tuner_errors(tmp_dir):
    # invalid oracle
    with pytest.raises(
        ValueError, match="Expected oracle to be an instance of Oracle"
    ):
        tuner_module.Tuner(
            oracle="invalid", hypermodel=build_model, directory=tmp_dir
        )
    # invalid hypermodel
    with pytest.raises(ValueError, match="`hypermodel` argument should be either"):
        tuner_module.Tuner(
            oracle=keras_tuner.tuners.randomsearch.RandomSearchOracle(
                objective="val_accuracy", max_trials=3
            ),
            hypermodel="build_model",
            directory=tmp_dir,
        )
    # oversize model
    with pytest.raises(RuntimeError, match="Too many consecutive oversized models"):
        tuner = tuner_module.Tuner(
            oracle=keras_tuner.tuners.randomsearch.RandomSearchOracle(
                objective="val_accuracy", max_trials=3
            ),
            hypermodel=build_model,
            max_model_size=4,
            directory=tmp_dir,
        )
        tuner.search(
            TRAIN_INPUTS, TRAIN_TARGETS, validation_data=(VAL_INPUTS, VAL_TARGETS)
        )
    # TODO: test no optimizer


@pytest.mark.skipif(
    parse(tf.__version__) < parse("2.3.0"),
    reason="TPUStrategy only exists in TF2.3+.",
)
def test_metric_direction_inferred_from_objective(tmp_dir):
    oracle = keras_tuner.tuners.randomsearch.RandomSearchOracle(
        objective=keras_tuner.Objective("a", "max"), max_trials=1
    )
    oracle._set_project_dir(tmp_dir, "untitled_project")
    trial = oracle.create_trial("tuner0")
    oracle.update_trial(trial.trial_id, {"a": 1})
    trial = oracle.get_trial(trial.trial_id)
    assert trial.metrics.get_direction("a") == "max"

    oracle = keras_tuner.tuners.randomsearch.RandomSearchOracle(
        objective=keras_tuner.Objective("a", "min"), max_trials=1
    )
    oracle._set_project_dir(tmp_dir, "untitled_project2")
    trial = oracle.create_trial("tuner0")
    oracle.update_trial(trial.trial_id, {"a": 1})
    trial = oracle.get_trial(trial.trial_id)
    assert trial.metrics.get_direction("a") == "min"


def test_overwrite_true(tmp_dir):
    tuner = keras_tuner.tuners.RandomSearch(
        hypermodel=build_model,
        objective="val_accuracy",
        max_trials=2,
        directory=tmp_dir,
    )
    tuner.search(
        TRAIN_INPUTS, TRAIN_TARGETS, validation_data=(VAL_INPUTS, VAL_TARGETS)
    )
    assert len(tuner.oracle.trials) == 2

    new_tuner = keras_tuner.tuners.RandomSearch(
        hypermodel=build_model,
        objective="val_accuracy",
        max_trials=2,
        directory=tmp_dir,
        overwrite=True,
    )
    assert len(new_tuner.oracle.trials) == 0


def test_correct_display_trial_number(tmp_dir):
    tuner = keras_tuner.tuners.RandomSearch(
        hypermodel=build_model,
        objective="val_accuracy",
        max_trials=2,
        directory=tmp_dir,
    )
    tuner.search(
        TRAIN_INPUTS, TRAIN_TARGETS, validation_data=(VAL_INPUTS, VAL_TARGETS)
    )
    new_tuner = keras_tuner.tuners.RandomSearch(
        hypermodel=build_model,
        objective="val_accuracy",
        max_trials=6,
        directory=tmp_dir,
        overwrite=False,
    )
    new_tuner.search(
        TRAIN_INPUTS, TRAIN_TARGETS, validation_data=(VAL_INPUTS, VAL_TARGETS)
    )
    assert len(new_tuner.oracle.trials) == new_tuner._display.trial_number


def test_error_on_unknown_objective_direction(tmp_dir):
    with pytest.raises(ValueError, match="Could not infer optimization direction"):
        keras_tuner.tuners.RandomSearch(
            hypermodel=build_model,
            objective="custom_metric",
            max_trials=2,
            directory=tmp_dir,
        )


def test_callbacks_run_each_execution(tmp_dir):
    callback_instances = set()

    class LoggingCallback(keras.callbacks.Callback):
        def on_train_begin(self, logs):
            callback_instances.add(id(self))

    logging_callback = LoggingCallback()
    tuner = keras_tuner.tuners.RandomSearch(
        hypermodel=build_model,
        objective="val_accuracy",
        max_trials=2,
        executions_per_trial=3,
        directory=tmp_dir,
    )
    tuner.search(
        TRAIN_INPUTS,
        TRAIN_TARGETS,
        validation_data=(VAL_INPUTS, VAL_TARGETS),
        callbacks=[logging_callback],
    )

    assert len(callback_instances) == 6


def test_build_and_fit_model_in_multi_execution_tuner(tmp_dir):
    class MyTuner(keras_tuner.tuners.RandomSearch):
        def _build_and_fit_model(self, trial, fit_args, fit_kwargs):
            self.was_called = True
            return super()._build_and_fit_model(trial, fit_args, fit_kwargs)

    tuner = MyTuner(
        hypermodel=build_model,
        objective="val_accuracy",
        max_trials=2,
        executions_per_trial=3,
        directory=tmp_dir,
    )

    tuner.run_trial(
        tuner.oracle.create_trial("tuner0"),
        TRAIN_INPUTS,
        TRAIN_TARGETS,
        validation_data=(VAL_INPUTS, VAL_TARGETS),
    )

    assert tuner.was_called


def test_build_and_fit_model_in_tuner(tmp_dir):
    class MyTuner(tuner_module.Tuner):
        def _build_and_fit_model(self, trial, fit_args, fit_kwargs):
            self.was_called = True
            return super()._build_and_fit_model(trial, fit_args, fit_kwargs)

    tuner = MyTuner(
        oracle=keras_tuner.tuners.randomsearch.RandomSearchOracle(
            objective="val_loss",
            max_trials=2,
        ),
        hypermodel=build_model,
        directory=tmp_dir,
    )

    tuner.run_trial(
        tuner.oracle.create_trial("tuner0"),
        TRAIN_INPUTS,
        TRAIN_TARGETS,
        validation_data=(VAL_INPUTS, VAL_TARGETS),
    )

    assert tuner.was_called


def test_init_build_all_hps_in_all_conditions(tmp_dir):
    class ConditionalHyperModel(MockHyperModel):
        def build(self, hp):
            model_type = hp.Choice("model_type", ["cnn", "mlp"])
            with hp.conditional_scope("model_type", ["cnn"]):
                if model_type == "cnn":
                    sub_cnn = hp.Choice("sub_cnn", ["a", "b"])
                    with hp.conditional_scope("sub_cnn", ["a"]):
                        if sub_cnn == "a":
                            hp.Int("n_filters_a", 2, 4)
                    with hp.conditional_scope("sub_cnn", ["b"]):
                        if sub_cnn == "b":
                            hp.Int("n_filters_b", 6, 8)
            with hp.conditional_scope("model_type", ["mlp"]):
                if model_type == "mlp":
                    sub_mlp = hp.Choice("sub_mlp", ["a", "b"])
                    with hp.conditional_scope("sub_mlp", ["a"]):
                        if sub_mlp == "a":
                            hp.Int("n_units_a", 2, 4)
                    with hp.conditional_scope("sub_mlp", ["b"]):
                        if sub_mlp == "b":
                            hp.Int("n_units_b", 6, 8)
            more_block = hp.Boolean("more_block", default=False)
            with hp.conditional_scope("more_block", [True]):
                if more_block:
                    hp.Int("new_block_hp", 1, 3)
            return super().build(hp)

    def name_in_hp(name, hp):
        return any([name == single_hp.name for single_hp in hp.space])

    class MyTuner(tuner_module.Tuner):
        def _populate_initial_space(self):
            super()._populate_initial_space()
            hp = self.oracle.hyperparameters
            assert name_in_hp("model_type", hp)
            assert name_in_hp("sub_cnn", hp)
            assert name_in_hp("n_filters_a", hp)
            assert name_in_hp("n_filters_b", hp)
            assert name_in_hp("sub_mlp", hp)
            assert name_in_hp("n_units_a", hp)
            assert name_in_hp("n_units_b", hp)
            assert name_in_hp("more_block", hp)
            assert name_in_hp("new_block_hp", hp)

    MyTuner(
        oracle=keras_tuner.tuners.randomsearch.RandomSearchOracle(
            objective="loss", max_trials=2, seed=1337
        ),
        hypermodel=ConditionalHyperModel(),
        directory=tmp_dir,
    )
