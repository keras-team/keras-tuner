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

import logging
import numpy as np
import pytest
import sys
import tensorflow as tf

import kerastuner as kt
from kerastuner.tuners import hyperband as hyperband_module


@pytest.fixture(scope='function')
def tmp_dir(tmpdir_factory):
    return tmpdir_factory.mktemp('hyperband_test', numbered=True)


def build_model(hp):
    model = tf.keras.Sequential()
    for i in range(hp.Int('layers', 1, 3)):
        model.add(tf.keras.layers.Dense(
            hp.Int('units' + str(i), 1, 5),
            activation='relu'))
        model.add(tf.keras.layers.Lambda(
            lambda x: x + hp.Float('bias' + str(i), -1, 1)))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    model.compile('sgd', 'mse')
    return model


def test_hyperband_oracle_bracket_configs(tmp_dir):
    oracle = hyperband_module.HyperbandOracle(
        objective=kt.Objective('score', 'max'),
        hyperband_iterations=1,
        max_epochs=8,
        factor=2)
    oracle._set_project_dir(tmp_dir, 'untitled')

    # 8, 4, 2, 1 starting epochs.
    assert oracle._get_num_brackets() == 4

    assert oracle._get_num_rounds(bracket_num=3) == 4
    assert oracle._get_size(bracket_num=3, round_num=0) == 8
    assert oracle._get_epochs(bracket_num=3, round_num=0) == 1
    assert oracle._get_size(bracket_num=3, round_num=3) == 1
    assert oracle._get_epochs(bracket_num=3, round_num=3) == 8

    assert oracle._get_num_rounds(bracket_num=0) == 1
    assert oracle._get_size(bracket_num=0, round_num=0) == 4
    assert oracle._get_epochs(bracket_num=0, round_num=0) == 8


@pytest.mark.skipif(sys.version_info < (3, 0), reason='TODO: Enable test for Py2')
def test_hyperband_oracle_one_sweep_single_thread(tmp_dir):
    hp = kt.HyperParameters()
    hp.Float('a', -100, 100)
    hp.Float('b', -100, 100)
    oracle = hyperband_module.HyperbandOracle(
        hyperparameters=hp,
        objective=kt.Objective('score', 'max'),
        hyperband_iterations=1,
        max_epochs=9,
        factor=3)
    oracle._set_project_dir(tmp_dir, 'untitled')

    score = 0
    for bracket_num in reversed(range(oracle._get_num_brackets())):
        for round_num in range(oracle._get_num_rounds(bracket_num)):
            for model_num in range(oracle._get_size(bracket_num, round_num)):
                trial = oracle.create_trial('tuner0')
                assert trial.status == 'RUNNING'
                score += 1
                oracle.update_trial(
                    trial.trial_id,
                    {'score': score})
                oracle.end_trial(
                    trial.trial_id,
                    status='COMPLETED')
            assert len(oracle._brackets[0]['rounds'][round_num]) == oracle._get_size(
                bracket_num, round_num)
        assert len(oracle._brackets) == 1

    # Iteration should now be complete.
    trial = oracle.create_trial('tuner0')
    assert trial.status == 'STOPPED', oracle.hyperband_iterations
    assert len(oracle.ongoing_trials) == 0

    # Brackets should all be finished and removed.
    assert len(oracle._brackets) == 0

    best_trial = oracle.get_best_trials()[0]
    assert best_trial.score == score


def test_hyperband_oracle_one_sweep_parallel(tmp_dir):
    hp = kt.HyperParameters()
    hp.Float('a', -100, 100)
    hp.Float('b', -100, 100)
    oracle = hyperband_module.HyperbandOracle(
        hyperparameters=hp,
        objective=kt.Objective('score', 'max'),
        hyperband_iterations=1,
        max_epochs=4,
        factor=2)
    oracle._set_project_dir(tmp_dir, 'untitled')

    # All round 0 trials from different brackets can be run
    # in parallel.
    round0_trials = []
    for i in range(10):
        t = oracle.create_trial('tuner' + str(i))
        assert t.status == 'RUNNING'
        round0_trials.append(t)

    assert len(oracle._brackets) == 3

    # Round 1 can't be run until enough models from round 0
    # have completed.
    t = oracle.create_trial('tuner10')
    assert t.status == 'IDLE'

    for t in round0_trials:
        oracle.update_trial(t.trial_id, {'score': 1})
        oracle.end_trial(t.trial_id, 'COMPLETED')

    round1_trials = []
    for i in range(4):
        t = oracle.create_trial('tuner' + str(i))
        assert t.status == 'RUNNING'
        round1_trials.append(t)

    # Bracket 0 is complete as it only has round 0.
    assert len(oracle._brackets) == 2

    # Round 2 can't be run until enough models from round 1
    # have completed.
    t = oracle.create_trial('tuner10')
    assert t.status == 'IDLE'

    for t in round1_trials:
        oracle.update_trial(t.trial_id, {'score': 1})
        oracle.end_trial(t.trial_id, 'COMPLETED')

    # Only one trial runs in round 2.
    round2_trial = oracle.create_trial('tuner0')

    assert len(oracle._brackets) == 1

    # No more trials to run, but wait for existing brackets to end.
    t = oracle.create_trial('tuner10')
    assert t.status == 'IDLE'

    oracle.update_trial(round2_trial.trial_id, {'score': 1})
    oracle.end_trial(round2_trial.trial_id, 'COMPLETED')

    t = oracle.create_trial('tuner10')
    assert t.status == 'STOPPED', oracle._current_sweep


@pytest.mark.skipif(sys.version_info < (3, 0), reason='TODO: Enable test for Py2')
def test_hyperband_integration(tmp_dir):
    tuner = hyperband_module.Hyperband(
        objective='val_loss',
        hypermodel=build_model,
        hyperband_iterations=2,
        max_epochs=6,
        factor=3,
        directory=tmp_dir)

    x, y = np.ones((2, 5)), np.ones((2, 1))
    tuner.search(x, y, validation_data=(x, y))

    # Make sure Oracle is registering new HPs.
    updated_hps = tuner.oracle.get_space().values
    assert 'units1' in updated_hps
    assert 'bias1' in updated_hps

    tf.get_logger().setLevel(logging.ERROR)

    best_score = tuner.oracle.get_best_trials()[0].score
    best_model = tuner.get_best_models()[0]
    assert best_model.evaluate(x, y) == best_score


@pytest.mark.skipif(sys.version_info < (3, 0), reason='TODO: Enable test for Py2')
def test_hyperband_save_and_restore(tmp_dir):
    tuner = hyperband_module.Hyperband(
        objective='val_loss',
        hypermodel=build_model,
        hyperband_iterations=1,
        max_epochs=7,
        factor=2,
        directory=tmp_dir)

    x, y = np.ones((2, 5)), np.ones((2, 1))
    tuner.search(x, y, validation_data=(x, y))

    num_trials = len(tuner.oracle.trials)
    assert num_trials > 0
    assert tuner.oracle._current_iteration == 1

    tuner.save()
    tuner.trials = {}
    tuner.oracle._current_iteration = 0
    tuner.reload()

    assert len(tuner.oracle.trials) == num_trials
    assert tuner.oracle._current_iteration == 1
