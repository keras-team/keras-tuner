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
import os
import pytest
import tensorflow as tf

from kerastuner.engine import hyperparameters
from kerastuner.engine import trial as trial_module
from kerastuner.engine import hyperparameters as hp_module
from kerastuner.tuners import hyperband as hyperband_module


@pytest.fixture(scope='module')
def tmp_dir(tmpdir_factory):
    return tmpdir_factory.mktemp('hyperband_test')


def test_hyperband_oracle(tmp_dir):
    hp_list = [hp_module.Choice('a', [1, 2], default=1),
               hp_module.Choice('b', [3, 4], default=3),
               hp_module.Choice('c', [5, 6], default=5),
               hp_module.Choice('d', [7, 8], default=7),
               hp_module.Choice('e', [9, 0], default=9)]
    oracle = hyperband_module.HyperbandOracle()
    assert oracle._num_brackets == 3

    for trial_id in range(oracle._model_sequence[0]):
        hp = oracle.populate_space('0_' + str(trial_id), hp_list)
        assert hp['status'] == 'RUN'
        assert hp['values']['tuner/epochs'] == oracle._epoch_sequence[0]
        assert 'tuner/trial_id' not in hp['values']
    assert oracle.populate_space('idle0', hp_list)['status'] == 'IDLE'
    for trial_id in range(oracle._model_sequence[0]):
        oracle.result('0_' + str(trial_id), trial_id)

    for trial_id in range(oracle._model_sequence[1]):
        hp = oracle.populate_space('1_' + str(trial_id), hp_list)
        assert hp['status'] == 'RUN'
        assert hp['values']['tuner/epochs'] == oracle._epoch_sequence[1]
        assert 'tuner/trial_id' in hp['values']
    assert oracle.populate_space('idle1', hp_list)['status'] == 'IDLE'
    for trial_id in range(oracle._model_sequence[1]):
        oracle.result('1_' + str(trial_id), trial_id)

    for trial_id in range(oracle._model_sequence[2]):
        hp = oracle.populate_space('2_' + str(trial_id), hp_list)
        assert hp['status'] == 'RUN'
        assert hp['values']['tuner/epochs'] == oracle._epoch_sequence[2]
        assert 'tuner/trial_id' in hp['values']
    assert oracle.populate_space('idle2', hp_list)['status'] == 'IDLE'
    for trial_id in range(oracle._model_sequence[2]):
        oracle.result('2_' + str(trial_id), trial_id)

    for trial_id in range(oracle._model_sequence[0]):
        hp = oracle.populate_space('3_' + str(trial_id), hp_list)
        assert hp['status'] == 'RUN'
        assert hp['values']['tuner/epochs'] == oracle._epoch_sequence[0]
        assert 'tuner/trial_id' not in hp['values']
    assert oracle.populate_space('idle3', hp_list)['status'] == 'IDLE'
    for trial_id in range(oracle._model_sequence[0]):
        oracle.result('3_' + str(trial_id), trial_id)

    assert oracle.populate_space('last', hp_list)['status'] == 'RUN'


def test_hyperband_dynamic_space(tmp_dir):
    hp_list = [hp_module.Choice('a', [1, 2], default=1)]
    oracle = hyperband_module.HyperbandOracle()
    hp_list.append(hp_module.Choice('b', [3, 4], default=3))
    values = oracle.populate_space('0', hp_list)['values']
    assert 'b' in values
    oracle.update_space(hp_list)
    hp_list.append(hp_module.Choice('c', [5, 6], default=5))
    assert 'c' in oracle.populate_space('1', hp_list)['values']
    hp_list.append(hp_module.Choice('d', [7, 8], default=7))
    assert 'd' in oracle.populate_space('2', hp_list)['values']
    hp_list.append(hp_module.Choice('e', [9, 0], default=9))
    assert 'e' in oracle.populate_space('3', hp_list)['values']


def test_hyperband_save_load_middle_of_bracket(tmp_dir):
    hp_list = [hp_module.Choice('a', [1, 2], default=1),
               hp_module.Choice('b', [3, 4], default=3),
               hp_module.Choice('c', [5, 6], default=5),
               hp_module.Choice('d', [7, 8], default=7),
               hp_module.Choice('e', [9, 0], default=9)]
    oracle = hyperband_module.HyperbandOracle()

    for trial_id in range(3):
        oracle.populate_space('0_' + str(trial_id), hp_list)
    for trial_id in range(2):
        oracle.result('0_' + str(trial_id), trial_id)

    fname = os.path.join(tmp_dir, 'oracle')
    oracle.save(fname)
    oracle = hyperband_module.HyperbandOracle()
    oracle.reload(fname)

    for trial_id in range(oracle._model_sequence[0] - 2):
        hp = oracle.populate_space('1_' + str(trial_id), hp_list)
        assert hp['status'] == 'RUN'
    assert oracle.populate_space('idle', hp_list)['status'] == 'IDLE'
    for trial_id in range(oracle._model_sequence[0] - 2):
        oracle.result('1_' + str(trial_id), trial_id)


def test_hyperband_save_load_at_begining(tmp_dir):
    hp_list = [hp_module.Choice('a', [1, 2], default=1),
               hp_module.Choice('b', [3, 4], default=3),
               hp_module.Choice('c', [5, 6], default=5),
               hp_module.Choice('d', [7, 8], default=7),
               hp_module.Choice('e', [9, 0], default=9)]
    oracle = hyperband_module.HyperbandOracle()

    fname = os.path.join(tmp_dir, 'oracle')
    oracle.save(fname)
    oracle = hyperband_module.HyperbandOracle()
    oracle.reload(fname)

    for trial_id in range(oracle._model_sequence[0]):
        hp = oracle.populate_space('0_' + str(trial_id), hp_list)
        assert hp['status'] == 'RUN'
    assert oracle.populate_space('idle', hp_list)['status'] == 'IDLE'
    for trial_id in range(oracle._model_sequence[0]):
        oracle.result('0_' + str(trial_id), trial_id)


def test_hyperband_save_load_at_the_end_of_bracket(tmp_dir):
    hp_list = [hp_module.Choice('a', [1, 2], default=1),
               hp_module.Choice('b', [3, 4], default=3),
               hp_module.Choice('c', [5, 6], default=5),
               hp_module.Choice('d', [7, 8], default=7),
               hp_module.Choice('e', [9, 0], default=9)]
    oracle = hyperband_module.HyperbandOracle()

    for trial_id in range(oracle._model_sequence[0]):
        hp = oracle.populate_space('0_' + str(trial_id), hp_list)
        assert hp['status'] == 'RUN'
    assert oracle.populate_space('idle', hp_list)['status'] == 'IDLE'
    for trial_id in range(oracle._model_sequence[0]):
        oracle.result('0_' + str(trial_id), trial_id)

    fname = os.path.join(tmp_dir, 'oracle')
    oracle.save(fname)
    oracle = hyperband_module.HyperbandOracle()
    oracle.reload(fname)

    for trial_id in range(oracle._model_sequence[1]):
        hp = oracle.populate_space('1_' + str(trial_id), hp_list)
        assert hp['status'] == 'RUN'
    assert oracle.populate_space('idle1', hp_list)['status'] == 'IDLE'


def test_hyperband_save_load_at_the_end_of_bandit(tmp_dir):
    hp_list = [hp_module.Choice('a', [1, 2], default=1),
               hp_module.Choice('b', [3, 4], default=3),
               hp_module.Choice('c', [5, 6], default=5),
               hp_module.Choice('d', [7, 8], default=7),
               hp_module.Choice('e', [9, 0], default=9)]
    oracle = hyperband_module.HyperbandOracle()

    for trial_id in range(oracle._model_sequence[0]):
        hp = oracle.populate_space('0_' + str(trial_id), hp_list)
        assert hp['status'] == 'RUN'
    assert oracle.populate_space('idle', hp_list)['status'] == 'IDLE'
    for trial_id in range(oracle._model_sequence[0]):
        oracle.result('0_' + str(trial_id), trial_id)

    for trial_id in range(oracle._model_sequence[1]):
        hp = oracle.populate_space('1_' + str(trial_id), hp_list)
        assert hp['status'] == 'RUN'
    assert oracle.populate_space('idle1', hp_list)['status'] == 'IDLE'
    for trial_id in range(oracle._model_sequence[1]):
        oracle.result('1_' + str(trial_id), trial_id)

    for trial_id in range(oracle._model_sequence[2]):
        hp = oracle.populate_space('2_' + str(trial_id), hp_list)
        assert hp['status'] == 'RUN'
        assert hp['values']['tuner/epochs'] == oracle._epoch_sequence[2]
        assert 'tuner/trial_id' in hp['values']
    assert oracle.populate_space('idle2', hp_list)['status'] == 'IDLE'
    for trial_id in range(oracle._model_sequence[2]):
        oracle.result('2_' + str(trial_id), trial_id)

    fname = os.path.join(tmp_dir, 'oracle')
    oracle.save(fname)
    oracle = hyperband_module.HyperbandOracle()
    oracle.reload(fname)

    for trial_id in range(oracle._model_sequence[0]):
        hp = oracle.populate_space('3_' + str(trial_id), hp_list)
        assert hp['status'] == 'RUN'
        assert hp['values']['tuner/epochs'] == oracle._epoch_sequence[0]
        assert 'tuner/trial_id' not in hp['values']
    assert oracle.populate_space('idle3', hp_list)['status'] == 'IDLE'


def build_model(hp):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=(2, 2)))
    for i in range(3):
        model.add(tf.keras.layers.Dense(units=hp.Int('units_' + str(i),
                                                     2, 4, 2),
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


class HyperbandStub(hyperband_module.Hyperband):
    def on_execution_end(self, trial, execution, model):
        pass

    def __init__(self, hypermodel, objective, max_trials, **kwargs):
        super().__init__(hypermodel, objective, max_trials, **kwargs)
        hp = hyperparameters.HyperParameters()
        trial = trial_module.Trial('1', hp, 5, base_directory=self.directory)
        trial.executions = [
            execution_module.Execution('a', 'b', 1, 3,
                                       base_directory=self.directory)]
        trial.executions[0].best_checkpoint = 'x'
        self.trials = [trial]


@mock.patch('tensorflow.keras.Model.fit', side_effect=mock_fit)
@mock.patch('tensorflow.keras.Model.load_weights', side_effect=mock_load)
def test_hyperband_tuner(patch_fit, patch_load, tmp_dir):
    x = np.random.rand(10, 2, 2).astype('float32')
    y = np.random.randint(0, 1, (10,))
    val_x = np.random.rand(10, 2, 2).astype('float32')
    val_y = np.random.randint(0, 1, (10,))

    tuner = HyperbandStub(
        build_model,
        objective='val_accuracy',
        max_trials=15,
        factor=2,
        min_epochs=1,
        max_epochs=2,
        executions_per_trial=3,
        directory=tmp_dir)

    hp = hyperparameters.HyperParameters()
    hp.values['tuner/epochs'] = 10
    trial_id = '1'
    hp.values['tuner/trial_id'] = trial_id

    tuner.run_trial(
        trial_module.Trial(trial_id, hp, 5, base_directory=tmp_dir), hp, [],
        {'x': x, 'y': y, 'epochs': 1, 'validation_data': (val_x, val_y)})
    assert patch_fit.called
    assert patch_load.called
