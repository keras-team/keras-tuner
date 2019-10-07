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


@pytest.fixture(scope='function')
def tmp_dir(tmpdir_factory):
    return tmpdir_factory.mktemp('hyperband_test', numbered=True)


def test_hyperband_oracle(tmp_dir):
    hps = hp_module.HyperParameters()

    hps.Choice('a', [1, 2], default=1)
    hps.Choice('b', [3, 4], default=3)
    hps.Choice('c', [5, 6], default=5)
    hps.Choice('d', [7, 8], default=7)
    hps.Choice('e', [9, 0], default=9)
    oracle = hyperband_module.HyperbandOracle(
        objective='score', max_trials=100, hyperparameters=hps)
    oracle.set_project_dir(tmp_dir, 'untitled', reload=False)
    assert oracle._num_brackets == 3
    assert len(oracle.trials) == 0

    for bracket in range(oracle._num_brackets):
        trials = []
        for i in range(oracle._model_sequence[bracket]):
            trial = oracle.create_trial(i)
            trials.append(trial)
            hp = trial.hyperparameters
            assert trial.status == 'RUNNING', i
            assert (hp.values['tuner/epochs'] ==
                    oracle._epoch_sequence[bracket])
            if bracket > 0:
                assert 'tuner/trial_id' in hp.values
            else:
                assert 'tuner/trial_id' not in hp.values

        # Asking for more trials when bracket is not yet complete.
        trial = oracle.create_trial('idle0')
        assert trial.status == 'IDLE'

        for trial in trials:
            oracle.update_trial(trial.trial_id, {'score': 1.})
            oracle.end_trial(trial.trial_id, 'COMPLETED')


def test_hyperband_dynamic_space(tmp_dir):
    hps = hp_module.HyperParameters()
    hps.Choice('a', [1, 2], default=1)
    oracle = hyperband_module.HyperbandOracle(
        objective='score', max_trials=50, hyperparameters=hps)
    oracle.set_project_dir(tmp_dir, 'untitled')
    hps.Choice('b', [3, 4], default=3)
    values = oracle._populate_space('0')['values']
    assert 'b' in values
    new_hps = hp_module.HyperParameters()
    new_hps.Choice('c', [5, 6], default=5)
    oracle.update_space(new_hps)
    assert 'c' in oracle._populate_space('1')['values']
    new_hps.Choice('d', [7, 8], default=7)
    oracle.update_space(new_hps)
    assert 'd' in oracle._populate_space('2')['values']
    new_hps.Choice('e', [9, 0], default=9)
    oracle.update_space(new_hps)
    assert 'e' in oracle._populate_space('3')['values']


def test_hyperband_save_load_middle_of_bracket(tmp_dir):
    hps = hp_module.HyperParameters()
    hps.Choice('a', [1, 2], default=1)
    hps.Choice('b', [3, 4], default=3)
    hps.Choice('c', [5, 6], default=5)
    hps.Choice('d', [7, 8], default=7)
    hps.Choice('e', [9, 0], default=9)
    oracle = hyperband_module.HyperbandOracle(
        objective='score', max_trials=50, hyperparameters=hps)
    oracle.set_project_dir(tmp_dir, 'untitled')

    trials = []
    for i in range(3):
        trial = oracle.create_trial(i)
        trials.append(trial)

    for i in range(2):
        trial = trials[i]
        oracle.update_trial(trial.trial_id, {'score': 1.})
        oracle.end_trial(trial.trial_id, "COMPLETED")

    oracle.save()
    oracle = hyperband_module.HyperbandOracle(
        objective='score', max_trials=50, hyperparameters=hps)
    oracle.reload()

    trials = []
    for i in range(oracle._model_sequence[0] - 2):
        trial = oracle.create_trial(i + 2)
        trials.append(trial)
        assert trial.status == 'RUNNING'

    # Asking for more trials when bracket is not yet complete.
    trial = oracle.create_trial('idle0')
    assert trial.status == 'IDLE'

    for trial in trials:
        oracle.update_trial(trial.trial_id, {'score': 1.})
        oracle.end_trial(trial.trial_id, 'COMPLETED')


def test_hyperband_save_load_at_begining(tmp_dir):
    hps = hp_module.HyperParameters()
    hps.Choice('a', [1, 2], default=1)
    hps.Choice('b', [3, 4], default=3)
    hps.Choice('c', [5, 6], default=5)
    hps.Choice('d', [7, 8], default=7)
    hps.Choice('e', [9, 0], default=9)
    oracle = hyperband_module.HyperbandOracle(
        objective='score', max_trials=50, hyperparameters=hps)
    oracle.set_project_dir(tmp_dir, 'untitled')

    oracle.save()
    oracle = hyperband_module.HyperbandOracle(
        objective='score', max_trials=50, hyperparameters=hps)
    oracle.set_project_dir(tmp_dir, 'untitled')
    oracle.reload()

    trials = []
    for i in range(oracle._model_sequence[0]):
        trial = oracle.create_trial(i)
        trials.append(trial)
        assert trial.status == 'RUNNING'
        oracle.update_trial(trial.trial_id, {'score': 1})

    trial = oracle.create_trial('idle0')
    assert trial.status == 'IDLE'

    for trial in trials:
        oracle.end_trial(trial.trial_id, 'COMPLETED')


def test_hyperband_save_load_at_the_end_of_bracket(tmp_dir):
    hps = hp_module.HyperParameters()
    hps.Choice('a', [1, 2], default=1)
    hps.Choice('b', [3, 4], default=3)
    hps.Choice('c', [5, 6], default=5)
    hps.Choice('d', [7, 8], default=7)
    hps.Choice('e', [9, 0], default=9)
    oracle = hyperband_module.HyperbandOracle(
        objective='score', max_trials=50, hyperparameters=hps)
    oracle.set_project_dir(tmp_dir, 'untitled3')

    trials = []
    for i in range(oracle._model_sequence[0]):
        trial = oracle.create_trial(i)
        trials.append(trial)
        assert trial.status == 'RUNNING'
        oracle.update_trial(trial.trial_id, {'score': 1})

    trial = oracle.create_trial('idle0')
    assert trial.status == 'IDLE'

    for trial in trials:
        oracle.end_trial(trial.trial_id, 'COMPLETED')

    fname = os.path.join(tmp_dir, 'oracle')
    oracle.save(fname)
    oracle = hyperband_module.HyperbandOracle(
        objective='score', max_trials=50, hyperparameters=hps)
    oracle.reload(fname)

    trials = []
    for i in range(oracle._model_sequence[1]):
        trial = oracle.create_trial(i)
        trials.append(trial)
        assert trial.status == 'RUNNING'
        oracle.update_trial(trial.trial_id, {'score': 1})

    trial = oracle.create_trial('idle1')
    assert trial.status == 'IDLE'

    for trial in trials:
        oracle.end_trial(trial.trial_id, 'COMPLETED')


def test_hyperband_save_load_at_the_end_of_bandit(tmp_dir):
    hps = hp_module.HyperParameters()
    hps.Choice('a', [1, 2], default=1)
    hps.Choice('b', [3, 4], default=3)
    hps.Choice('c', [5, 6], default=5)
    hps.Choice('d', [7, 8], default=7)
    hps.Choice('e', [9, 0], default=9)
    oracle = hyperband_module.HyperbandOracle(
        objective='score', max_trials=50, hyperparameters=hps)
    oracle.set_project_dir(tmp_dir, 'untitled')

    for bracket in range(oracle._num_brackets):
        trials = []
        for i in range(oracle._model_sequence[bracket]):
            trial = oracle.create_trial(i)
            trials.append(trial)
            hp = trial.hyperparameters
            assert trial.status == 'RUNNING'
            assert (hp.values['tuner/epochs'] ==
                    oracle._epoch_sequence[bracket])
            if bracket > 0:
                assert 'tuner/trial_id' in hp.values
            else:
                assert 'tuner/trial_id' not in hp.values

        # Asking for more trials when bracket is not yet complete.
        trial = oracle.create_trial('idle0')
        assert trial.status == 'IDLE'

        for trial in trials:
            oracle.update_trial(trial.trial_id, {'score': 1.})
            oracle.end_trial(trial.trial_id, 'COMPLETED')

    fname = os.path.join(tmp_dir, 'oracle')
    oracle.save(fname)
    oracle = hyperband_module.HyperbandOracle(
        objective='score', max_trials=50, hyperparameters=hps)
    oracle.reload(fname)

    trials = []
    for i in range(oracle._model_sequence[0]):
        trial = oracle.create_trial(i)
        trials.append(trial)
        hp = trial.hyperparameters
        assert trial.status == 'RUNNING'
        assert (hp.values['tuner/epochs'] ==
                oracle._epoch_sequence[0])
        assert 'tuner/trial_id' not in hp.values

    # Asking for more trials when bracket is not yet complete.
    trial = oracle.create_trial('idle0')
    assert trial.status == 'IDLE'

    for trial in trials:
        oracle.update_trial(trial.trial_id, {'score': 1.})
        oracle.end_trial(trial.trial_id, 'COMPLETED')


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
    history = tf.keras.callbacks.History()
    history.history = {'val_accuracy': 0.5}
    return history


def mock_load(best_checkpoint):
    assert 'epoch_0' in best_checkpoint


@mock.patch('tensorflow.keras.Model.fit', side_effect=mock_fit)
@mock.patch('tensorflow.keras.Model.load_weights', side_effect=mock_load)
def test_hyperband_tuner(patch_fit, patch_load, tmp_dir):
    x = np.random.rand(10, 2, 2).astype('float32')
    y = np.random.randint(0, 1, (10,))
    val_x = np.random.rand(10, 2, 2).astype('float32')
    val_y = np.random.randint(0, 1, (10,))

    tuner = hyperband_module.Hyperband(
        build_model,
        objective='val_accuracy',
        max_trials=15,
        factor=2,
        min_epochs=1,
        max_epochs=2,
        directory=tmp_dir)

    hp = hyperparameters.HyperParameters()
    history_trial = trial_module.Trial(hyperparameters=hp.copy())
    history_trial.score = 1
    history_trial.best_step = 0
    hp.values['tuner/epochs'] = 10
    hp.values['tuner/trial_id'] = history_trial.trial_id
    tuner.oracle.trials[history_trial.trial_id] = history_trial

    trial = trial_module.Trial(hyperparameters=hp)
    tuner.oracle.trials[trial.trial_id] = trial
    tuner.run_trial(
        trial,
        x=x,
        y=y,
        epochs=1,
        validation_data=(val_x, val_y))
    assert patch_fit.called
    assert patch_load.called
