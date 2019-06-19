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
from unittest import mock

import numpy as np
import tensorflow as tf

import kerastuner
from kerastuner.engine import hyperparameters
from kerastuner.engine import execution as execution_module
from kerastuner.engine import trial as trial_module
from kerastuner.tuners.ultraband import UltraBand


def build_model(hp):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=(2, 2)))
    for i in range(3):
        model.add(tf.keras.layers.Dense(units=hp.Range('units_' + str(i), 2, 4, 2),
                                        activation='relu'))
    model.add(tf.keras.layers.Dense(2, activation='softmax'))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4])),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'])
    return model


def mock_fit(**kwargs):
    assert kwargs['epochs'] == 10


def mock_load(best_checkpoint):
    assert best_checkpoint == 'x-weights.h5'


class UltraBandStub(UltraBand):
    def on_execution_end(self, trial, execution, model):
        pass

    def __init__(self, hypermodel, objective, max_trials, **kwargs):
        super().__init__(hypermodel, objective, max_trials, **kwargs)
        hp = hyperparameters.HyperParameters()
        trial = trial_module.Trial('1', hp, 5)
        trial.executions = [execution_module.Execution('a', 'b', 1, 3)]
        trial.executions[0].best_checkpoint = 'x'
        self.trials = [trial]


@mock.patch('tensorflow.keras.Model.fit', side_effect=mock_fit)
@mock.patch('tensorflow.keras.Model.load_weights', side_effect=mock_load)
def test_ultraband_tuner(patch_fit, patch_load):
    x = np.random.rand(10, 2, 2).astype('float32')
    y = np.random.randint(0, 1, (10,))
    val_x = np.random.rand(10, 2, 2).astype('float32')
    val_y = np.random.randint(0, 1, (10,))

    tuner = UltraBandStub(
        build_model,
        objective='val_accuracy',
        max_trials=15,
        executions_per_trial=3,
        directory='test_dir')

    hp = hyperparameters.HyperParameters()
    hp.values['epochs'] = 10
    trial_id = '1'
    hp.values['trial_id'] = trial_id

    tuner.run_trial(trial_module.Trial(trial_id, hp, 5), hp, [], {'x': x,
                                                                  'y': y,
                                                                  'epochs': 1,
                                                                  'validation_data': (val_x, val_y)})
    assert patch_fit.called
    assert patch_load.called
