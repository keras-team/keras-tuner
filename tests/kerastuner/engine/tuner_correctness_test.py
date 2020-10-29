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

import pytest
import mock
import numpy as np

import kerastuner
from kerastuner.engine import tuner as tuner_module

import tensorflow as tf
from tensorflow import keras

INPUT_DIM = 2
NUM_CLASSES = 3
NUM_SAMPLES = 64
TRAIN_INPUTS = np.random.random(size=(NUM_SAMPLES, INPUT_DIM))
TRAIN_TARGETS = np.random.randint(0, NUM_CLASSES, size=(NUM_SAMPLES, 1))
VAL_INPUTS = np.random.random(size=(NUM_SAMPLES, INPUT_DIM))
VAL_TARGETS = np.random.randint(0, NUM_CLASSES, size=(NUM_SAMPLES, 1))


@pytest.fixture(scope='function')
def tmp_dir(tmpdir_factory):
    return tmpdir_factory.mktemp('integration_test', numbered=True)


def build_model(hp):
    model = keras.Sequential([
        keras.layers.Dense(hp.Int('units', 100, 1000, 100),
                           input_shape=(INPUT_DIM,),
                           activation='relu'),
        keras.layers.Dense(NUM_CLASSES, activation='softmax')
    ])
    model.compile('rmsprop',
                  'sparse_categorical_crossentropy',
                  metrics=['accuracy'])
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
        logs = {
            'loss': np.average(self.full_history[epoch])
        }
        for callback in self.callbacks:
            callback.on_epoch_end(epoch, logs=logs)

    def on_batch_begin(self, epoch, batch):
        for callback in self.callbacks:
            callback.on_batch_begin(batch, logs=None)

    def on_batch_end(self, epoch, batch):
        logs = {
            'loss': self.full_history[epoch][batch]
        }
        for callback in self.callbacks:
            callback.on_batch_end(epoch, logs=logs)

    def fit(self, *args, **kwargs):
        self.callbacks = kwargs['callbacks']
        for callback in self.callbacks:
            callback.model = self
        for epoch in range(len(self.full_history)):
            self.on_epoch_begin(epoch)
            for batch in range(len(self.full_history[epoch])):
                self.on_batch_begin(epoch, batch)
                self.on_batch_end(epoch, batch)
            self.on_epoch_end(epoch)

    def save_weights(self, fname, **kwargs):
        pass

    def get_config(self):
        return {}


class MockHyperModel(kerastuner.HyperModel):

    mode_0 = [[10, 9, 8], [7, 6, 5], [4, 3, 2]]
    mode_1 = [[13, 13, 13], [12, 12, 12], [11, 11, 11]]

    def __init__(self):
        # The first call to `build` in tuner __init__
        # will reset this to 0
        self.mode_0_execution_count = -1

    def build(self, hp):
        if hp.Choice('mode', [0, 1]) == 0:
            return MockModel(self.mode_0)
        return MockModel(self.mode_1)


def test_tuning_correctness(tmp_dir):
    tuner = kerastuner.Tuner(
        oracle=kerastuner.tuners.randomsearch.RandomSearchOracle(
            objective='loss',
            max_trials=2,
            seed=1337),
        hypermodel=MockHyperModel(),
        directory=tmp_dir,
    )
    tuner.search()
    assert len(tuner.oracle.trials) == 2

    m0_epochs = [float(np.average(x)) for x in MockHyperModel.mode_0]
    m1_epochs = [float(np.average(x)) for x in MockHyperModel.mode_1]

    # Score tracking correctness
    first_trial, second_trial = sorted(
        tuner.oracle.trials.values(), key=lambda t: t.score)
    assert first_trial.score == min(m0_epochs)
    assert second_trial.score == min(m1_epochs)
    assert (tuner.oracle.get_best_trials(1)[0].trial_id ==
            first_trial.trial_id)


def test_tuner_errors(tmp_dir):
    # invalid oracle
    with pytest.raises(
            ValueError,
            match='Expected oracle to be an instance of Oracle'):
        tuner_module.Tuner(
            oracle='invalid',
            hypermodel=build_model,
            directory=tmp_dir)
    # invalid hypermodel
    with pytest.raises(
            ValueError,
            match='`hypermodel` argument should be either'):
        tuner_module.Tuner(
            oracle=kerastuner.tuners.randomsearch.RandomSearchOracle(
                objective='val_accuracy',
                max_trials=3),
            hypermodel='build_model',
            directory=tmp_dir)
    # oversize model
    with pytest.raises(
            RuntimeError,
            match='Too many consecutive oversized models'):
        tuner = tuner_module.Tuner(
            oracle=kerastuner.tuners.randomsearch.RandomSearchOracle(
                objective='val_accuracy',
                max_trials=3),
            hypermodel=build_model,
            max_model_size=4,
            directory=tmp_dir)
        tuner.search(TRAIN_INPUTS, TRAIN_TARGETS,
                     validation_data=(VAL_INPUTS, VAL_TARGETS))
    # TODO: test no optimizer


def test_checkpoint_removal(tmp_dir):
    def build_model(hp):
        model = keras.Sequential([
            keras.layers.Dense(hp.Int('size', 5, 10)),
            keras.layers.Dense(1)])
        model.compile('sgd', 'mse', metrics=['accuracy'])
        return model

    tuner = kerastuner.Tuner(
        oracle=kerastuner.tuners.randomsearch.RandomSearchOracle(
            objective='val_accuracy',
            max_trials=1,
            seed=1337),
        hypermodel=build_model,
        directory=tmp_dir,
    )
    x, y = np.ones((1, 5)), np.ones((1, 1))
    tuner.search(x,
                 y,
                 validation_data=(x, y),
                 epochs=21)
    trial = list(tuner.oracle.trials.values())[0]
    trial_id = trial.trial_id
    assert tf.io.gfile.exists(tuner._get_checkpoint_fname(trial_id, 20))
    assert not tf.io.gfile.exists(tuner._get_checkpoint_fname(trial_id, 10))


def test_checkpoint_fname_tpu(tmp_dir):
    def build_model(hp):
        model = keras.Sequential([
            keras.layers.Dense(hp.Int('size', 5, 10)),
            keras.layers.Dense(1)])
        model.compile('sgd', 'mse', metrics=['accuracy'])
        return model

    strategy = mock.MagicMock(spec=tf.distribute.TPUStrategy)

    tuner = kerastuner.Tuner(
        oracle=kerastuner.tuners.randomsearch.RandomSearchOracle(
            objective='val_accuracy',
            max_trials=1,
            seed=1337),
        hypermodel=build_model,
        directory=tmp_dir,
        distribution_strategy=strategy,
    )
    assert tuner._get_checkpoint_fname(trial_id=0, epoch=20).endswith('.h5')


def test_checkpoint_fname_no_tpu(tmp_dir):
    def build_model(hp):
        model = keras.Sequential([
            keras.layers.Dense(hp.Int('size', 5, 10)),
            keras.layers.Dense(1)])
        model.compile('sgd', 'mse', metrics=['accuracy'])
        return model

    tuner = kerastuner.Tuner(
        oracle=kerastuner.tuners.randomsearch.RandomSearchOracle(
            objective='val_accuracy',
            max_trials=1,
            seed=1337),
        hypermodel=build_model,
        directory=tmp_dir,
    )
    assert not tuner._get_checkpoint_fname(trial_id=0, epoch=20).endswith('.h5')


def test_metric_direction_inferred_from_objective(tmp_dir):
    oracle = kerastuner.tuners.randomsearch.RandomSearchOracle(
        objective=kerastuner.Objective('a', 'max'),
        max_trials=1)
    oracle._set_project_dir(tmp_dir, 'untitled_project')
    trial = oracle.create_trial('tuner0')
    oracle.update_trial(trial.trial_id, {'a': 1})
    trial = oracle.get_trial(trial.trial_id)
    assert trial.metrics.get_direction('a') == 'max'

    oracle = kerastuner.tuners.randomsearch.RandomSearchOracle(
        objective=kerastuner.Objective('a', 'min'),
        max_trials=1)
    oracle._set_project_dir(tmp_dir, 'untitled_project2')
    trial = oracle.create_trial('tuner0')
    oracle.update_trial(trial.trial_id, {'a': 1})
    trial = oracle.get_trial(trial.trial_id)
    assert trial.metrics.get_direction('a') == 'min'


def test_overwrite_true(tmp_dir):
    tuner = kerastuner.tuners.RandomSearch(
        hypermodel=build_model,
        objective='val_accuracy',
        max_trials=2,
        directory=tmp_dir)
    tuner.search(TRAIN_INPUTS, TRAIN_TARGETS,
                 validation_data=(VAL_INPUTS, VAL_TARGETS))
    assert len(tuner.oracle.trials) == 2

    new_tuner = kerastuner.tuners.RandomSearch(
        hypermodel=build_model,
        objective='val_accuracy',
        max_trials=2,
        directory=tmp_dir,
        overwrite=True)
    assert len(new_tuner.oracle.trials) == 0


def test_error_on_unknown_objective_direction(tmp_dir):
    with pytest.raises(ValueError,
                       match='Could not infer optimization direction'):
        kerastuner.tuners.RandomSearch(
            hypermodel=build_model,
            objective='custom_metric',
            max_trials=2,
            directory=tmp_dir)


def test_callbacks_run_each_execution(tmp_dir):
    callback_instances = set()

    class LoggingCallback(keras.callbacks.Callback):

        def on_train_begin(self, logs):
            callback_instances.add(id(self))

    logging_callback = LoggingCallback()
    tuner = kerastuner.tuners.RandomSearch(
        hypermodel=build_model,
        objective='val_accuracy',
        max_trials=2,
        executions_per_trial=3,
        directory=tmp_dir)
    tuner.search(TRAIN_INPUTS, TRAIN_TARGETS,
                 validation_data=(VAL_INPUTS, VAL_TARGETS),
                 callbacks=[logging_callback])

    assert len(callback_instances) == 6


def test_build_and_fit_model_in_multi_execution_tuner(tmp_dir):

    class MyTuner(kerastuner.tuners.RandomSearch):
        def _build_and_fit_model(self, trial, fit_args, fit_kwargs):
            self.was_called = True
            return super()._build_and_fit_model(trial, fit_args, fit_kwargs)

    tuner = MyTuner(
        hypermodel=build_model,
        objective='val_accuracy',
        max_trials=2,
        executions_per_trial=3,
        directory=tmp_dir)

    tuner.run_trial(
        tuner.oracle.create_trial('tuner0'),
        TRAIN_INPUTS,
        TRAIN_TARGETS,
        validation_data=(VAL_INPUTS, VAL_TARGETS))

    assert tuner.was_called


def test_build_and_fit_model_in_tuner(tmp_dir):

    class MyTuner(tuner_module.Tuner):
        def _build_and_fit_model(self, trial, fit_args, fit_kwargs):
            self.was_called = True
            return super()._build_and_fit_model(trial, fit_args, fit_kwargs)

    tuner = MyTuner(
        oracle=kerastuner.tuners.randomsearch.RandomSearchOracle(
            objective='val_loss',
            max_trials=2,
        ),
        hypermodel=build_model,
        directory=tmp_dir)

    tuner.run_trial(
        tuner.oracle.create_trial('tuner0'),
        TRAIN_INPUTS,
        TRAIN_TARGETS,
        validation_data=(VAL_INPUTS, VAL_TARGETS))

    assert tuner.was_called


def save_model_setup_tuner(tmp_dir):
    class MyTuner(tuner_module.Tuner):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.was_called = False

        def _delete_checkpoint(self, trial_id, epoch):
            self.was_called = True

        def _checkpoint_model(self, model, trial_id, epoch):
            pass

    class MyOracle(kerastuner.engine.oracle.Oracle):
        def get_trial(self, trial_id):
            trial = unittest.mock.Mock()
            trial.metrics = unittest.mock.Mock()
            trial.metrics.get_best_step.return_value = 5
            return trial

    return MyTuner(
        oracle=MyOracle(objective='val_accuracy'),
        hypermodel=build_model,
        directory=tmp_dir)


def test_save_model_delete_not_called(tmp_dir):
    tuner = save_model_setup_tuner(tmp_dir)
    tuner.save_model('a', None, step=15)
    assert not tuner.was_called


def test_save_model_delete_called(tmp_dir):
    tuner = save_model_setup_tuner(tmp_dir)
    tuner.save_model('a', None, step=16)
    assert tuner.was_called
