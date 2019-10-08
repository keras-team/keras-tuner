import numpy as np
import pytest
import tensorflow as tf
import kerastuner as kt
from kerastuner.engine import hyperparameters as hp_module
from kerastuner.engine import trial as trial_module
from kerastuner.tuners import bayesian as bo_module


@pytest.fixture(scope='function')
def tmp_dir(tmpdir_factory):
    return tmpdir_factory.mktemp('bayesian_test', numbered=True)


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


def test_bayesian_oracle(tmp_dir):
    hps = hp_module.HyperParameters()
    hps.Choice('a', [1, 2], default=1)
    hps.Int('b', 3, 10, default=3)
    hps.Float('c', 0, 1, 0.1, default=0)
    hps.Fixed('d', 7)
    hps.Choice('e', [9, 0], default=9)
    oracle = bo_module.BayesianOptimizationOracle(
        objective='score', max_trials=20, hyperparameters=hps)
    oracle._set_project_dir(tmp_dir, 'untitled')
    for i in range(5):
        trial = oracle.create_trial(str(i))
        oracle.update_trial(trial.trial_id, {'score': i})
        oracle.end_trial(trial.trial_id, "COMPLETED")


def test_bayesian_oracle_with_zero_y(tmp_dir):
    hps = hp_module.HyperParameters()
    hps.Choice('a', [1, 2], default=1)
    hps.Int('b', 3, 10, default=3)
    hps.Float('c', 0, 1, 0.1, default=0)
    hps.Fixed('d', 7)
    hps.Choice('e', [9, 0], default=9)
    oracle = bo_module.BayesianOptimizationOracle(
        objective='score', max_trials=20, hyperparameters=hps)
    oracle._set_project_dir(tmp_dir, 'untitled')
    for i in range(5):
        trial = oracle.create_trial(str(i))
        oracle.update_trial(trial.trial_id, {'score': 0})
        oracle.end_trial(trial.trial_id, "COMPLETED")


def test_bayesian_dynamic_space(tmp_dir):
    hps = hp_module.HyperParameters()
    hps.Choice('a', [1, 2], default=1)
    oracle = bo_module.BayesianOptimizationOracle(
        objective='val_acc', max_trials=20)
    oracle._set_project_dir(tmp_dir, 'untitled')
    oracle.hyperparameters = hps
    for i in range(10):
        oracle._populate_space(str(i))
    hps.Int('b', 3, 10, default=3)
    assert 'b' in oracle._populate_space('1_0')['values']
    hps.Float('c', 0, 1, 0.1, default=0)
    assert 'c' in oracle._populate_space('1_1')['values']
    hps.Fixed('d', 7)
    assert 'd' in oracle._populate_space('1_2')['values']
    hps.Choice('e', [9, 0], default=9)
    assert 'e' in oracle._populate_space('1_3')['values']


def test_bayesian_save_reload(tmp_dir):
    hps = hp_module.HyperParameters()
    hps.Choice('a', [1, 2], default=1)
    hps.Choice('b', [3, 4], default=3)
    hps.Choice('c', [5, 6], default=5)
    hps.Choice('d', [7, 8], default=7)
    hps.Choice('e', [9, 0], default=9)
    oracle = bo_module.BayesianOptimizationOracle(
        objective='score', max_trials=20, hyperparameters=hps)
    oracle._set_project_dir(tmp_dir, 'untitled')

    for _ in range(3):
        trial = oracle.create_trial('tuner_id')
        oracle.update_trial(trial.trial_id, {'score': 1.})
        oracle.end_trial(trial.trial_id, "COMPLETED")

    oracle.save()
    oracle = bo_module.BayesianOptimizationOracle(
        objective='score', max_trials=20, hyperparameters=hps)
    oracle._set_project_dir(tmp_dir, 'untitled')
    oracle.reload()

    for trial_id in range(3):
        trial = oracle.create_trial('tuner_id')
        oracle.update_trial(trial.trial_id, {'score': 1.})
        oracle.end_trial(trial.trial_id, "COMPLETED")

    assert len(oracle.trials) == 6


def test_bayesian_optimization_tuner(tmp_dir):
    tuner = bo_module.BayesianOptimization(
        build_model,
        objective='val_accuracy',
        max_trials=15,
        directory=tmp_dir)
    assert isinstance(tuner.oracle, bo_module.BayesianOptimizationOracle)


def test_save_before_result(tmp_dir):
    hps = hp_module.HyperParameters()
    hps.Choice('a', [1, 2], default=1)
    hps.Int('b', 3, 10, default=3)
    hps.Float('c', 0, 1, 0.1, default=0)
    hps.Fixed('d', 7)
    hps.Choice('e', [9, 0], default=9)
    oracle = bo_module.BayesianOptimizationOracle(
        objective='score', max_trials=10, hyperparameters=hps)
    oracle._set_project_dir(tmp_dir, 'untitled')
    oracle._populate_space(str(1))
    oracle.save()


def test_bayesian_oracle_maximize(tmp_dir):
    hps = hp_module.HyperParameters()
    hps.Int('a', -100, 100)

    oracle = bo_module.BayesianOptimizationOracle(
        objective=kt.Objective('score', direction='max'),
        max_trials=20,
        hyperparameters=hps,
        num_initial_points=2)
    oracle._set_project_dir(tmp_dir, 'untitled')

    # Make examples with high 'a' and high score.
    for i in range(5):
        trial = trial_module.Trial(hyperparameters=hps.copy())
        trial.hyperparameters.values['a'] = 10*i
        trial.score = i
        trial.status = 'COMPLETED'
        oracle.trials[trial.trial_id] = trial

    # Make examples with low 'a' and low score
    for i in range(5):
        trial = trial_module.Trial(hyperparameters=hps.copy())
        trial.hyperparameters.values['a'] = -10*i
        trial.score = -i
        trial.status = 'COMPLETED'
        oracle.trials[trial.trial_id] = trial

    trial = oracle.create_trial('tuner0')
    assert trial.status == 'RUNNING'
    # Assert that the oracle suggests hps it thinks will maximize.
    assert trial.hyperparameters.get('a') > 0


def test_hyperparameters_added(tmp_dir):
    hps = hp_module.HyperParameters()
    hps.Int('a', -100, 100)

    oracle = bo_module.BayesianOptimizationOracle(
        objective=kt.Objective('score', direction='max'),
        max_trials=20,
        hyperparameters=hps,
        num_initial_points=2)
    oracle._set_project_dir(tmp_dir, 'untitled')

    # Populate initial trials.
    for i in range(10):
        trial = trial_module.Trial(hyperparameters=hps.copy())
        trial.hyperparameters.values['a'] = 10*i
        trial.score = i
        trial.status = 'COMPLETED'
        oracle.trials[trial.trial_id] = trial

    # Update the space.
    new_hps = hp_module.HyperParameters()
    new_hps.Float('b', 3.2, 6.4, step=0.2, default=3.6)
    new_hps.Boolean('c', default=True)
    oracle.update_space(new_hps)

    # Make sure fitting still works, with 'b' taking default in
    # prior trials.
    x, _ = oracle._vectorize_trials()
    assert np.allclose(x[:, 1], 3.6)
    assert np.allclose(x[:, 2], 1)

    # Make a new trial, it should have b set.
    trial = oracle.create_trial('tuner0')
    assert trial.status == 'RUNNING'
    assert 'b' in trial.hyperparameters.values
    assert 'c' in trial.hyperparameters.values
