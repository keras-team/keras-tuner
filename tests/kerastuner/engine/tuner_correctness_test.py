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

import pytest
import numpy as np

import kerastuner
from kerastuner.engine import tuner as tuner_module

from tensorflow import keras

INPUT_DIM = 2
NUM_CLASSES = 3
NUM_SAMPLES = 64
TRAIN_INPUTS = np.random.random(size=(NUM_SAMPLES, INPUT_DIM))
TRAIN_TARGETS = np.random.randint(0, NUM_CLASSES, size=(NUM_SAMPLES, 1))
VAL_INPUTS = np.random.random(size=(NUM_SAMPLES, INPUT_DIM))
VAL_TARGETS = np.random.randint(0, NUM_CLASSES, size=(NUM_SAMPLES, 1))


@pytest.fixture(scope='module')
def tmp_dir(tmpdir_factory):
    return tmpdir_factory.mktemp('integration_test')


def build_model(hp):
    model = keras.Sequential([
        keras.layers.Dense(hp.Range('units', 100, 1000, 100),
                           input_shape=(INPUT_DIM,),
                           activation='relu'),
        keras.layers.Dense(NUM_CLASSES, activation='softmax')
    ])
    model.compile('rmsprop', 'mse', metrics=['accuracy'])
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
            method(*args, *kwargs)

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
        f = open(fname, 'w')
        f.close()

    def get_config(self):
        return {}


class MockHyperModel(kerastuner.HyperModel):

    mode_0_execution_0 = [[10, 9, 8], [7, 6, 5], [4, 3, 2]]
    mode_0_execution_1 = [[12, 11, 10], [9, 8, 7], [6, 5, 4], [9, 9, 9]]
    mode_1_execution = [[13, 13, 13], [12, 12, 12], [11, 11, 11]]

    def __init__(self):
        self.mode_0_execution_count = 0

    def build(self, hp):
        if hp.Choice('mode', [0, 1]) == 0:
            self.mode_0_execution_count += 1
            if self.mode_0_execution_count == 1:
                return MockModel(self.mode_0_execution_0)
            else:
                return MockModel(self.mode_0_execution_1)
        return MockModel(self.mode_1_execution)


def test_tuning_correctness():
    tuner = kerastuner.RandomSearch(
        seed=1337,
        hypermodel=MockHyperModel(),
        max_trials=2,
        objective='loss',
        executions_per_trial=2,
    )
    tuner.search()
    assert len(tuner.trials) == 2

    m0_e0_epochs = [
        float(np.average(x)) for x in MockHyperModel.mode_0_execution_0]
    m0_e1_epochs = [
        float(np.average(x)) for x in MockHyperModel.mode_0_execution_1]
    m0_epochs = [
        (a + b) / 2
        for a, b in zip(m0_e0_epochs, m0_e1_epochs)] + [m0_e1_epochs[-1]]
    m1_epochs = [
        float(np.average(x)) for x in MockHyperModel.mode_1_execution]

    # Score tracking correctness
    first_trial, second_trial = sorted(
        tuner.trials, key=lambda x: x.score)
    assert first_trial.score == min(m0_epochs)
    assert second_trial.score == min(m1_epochs)
    status = tuner.get_status()
    assert status['best_trial']['score'] == min(m0_epochs)
    assert tuner._get_best_trials(1)[0] == first_trial

    # Metrics tracking correctness
    assert len(first_trial.executions) == 2
    e0 = first_trial.executions[0]
    e1 = first_trial.executions[1]
    e0_per_epoch_metrics = e0.per_epoch_metrics.metrics_history['loss']
    e1_per_epoch_metrics = e1.per_epoch_metrics.metrics_history['loss']
    e0_per_batch_metrics = e0.per_batch_metrics.metrics_history['loss']
    e1_per_batch_metrics = e1.per_batch_metrics.metrics_history['loss']
    assert e0_per_epoch_metrics == m0_e0_epochs
    assert e0_per_batch_metrics == MockHyperModel.mode_0_execution_0[-1]
    assert e1_per_epoch_metrics == m0_e1_epochs
    assert e1_per_batch_metrics == MockHyperModel.mode_0_execution_1[-1]
    assert first_trial.averaged_metrics.metrics_history['loss'] == m0_epochs


def test_tuner_errors():
    # invalid oracle
    with pytest.raises(
            ValueError,
            match='Expected oracle to be an instance of Oracle'):
        tuner_module.Tuner(
            oracle='invalid',
            hypermodel=build_model,
            objective='val_accuracy',
            max_trials=3)
    # invalid hypermodel
    with pytest.raises(
            ValueError,
            match='`hypermodel` argument should be either'):
        tuner_module.Tuner(
            oracle=kerastuner.tuners.randomsearch.RandomSearchOracle(),
            hypermodel='build_model',
            objective='val_accuracy',
            max_trials=3)
    # oversize model
    with pytest.raises(
            RuntimeError,
            match='Too many consecutive oversized models'):
        tuner = tuner_module.Tuner(
            oracle=kerastuner.tuners.randomsearch.RandomSearchOracle(),
            hypermodel=build_model,
            objective='val_accuracy',
            max_trials=3,
            max_model_size=4)
        tuner.search(TRAIN_INPUTS, TRAIN_TARGETS,
                     validation_data=(VAL_INPUTS, VAL_TARGETS))
    # TODO: test no optimizer
