import pytest
import os
import tensorflow as tf
from kerastuner.engine import hyperparameters as hp_module
from kerastuner.tuners import bayesian as bo_module


@pytest.fixture(scope='module')
def tmp_dir(tmpdir_factory):
    return tmpdir_factory.mktemp('bayesian_test')


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
    for i in range(5):
        trial = oracle.create_trial(str(i))
        oracle.update_trial(trial.trial_id, {'score': 0})
        oracle.end_trial(trial.trial_id, "COMPLETED")


def test_bayesian_dynamic_space(tmp_dir):
    hps = hp_module.HyperParameters()
    hps.Choice('a', [1, 2], default=1)
    oracle = bo_module.BayesianOptimizationOracle(
        objective='val_acc', max_trials=20)
    oracle.hyperparameters = hps
    for i in range(10):
        oracle.populate_space(str(i))
    hps.Int('b', 3, 10, default=3)
    assert 'b' in oracle.populate_space('1_0')['values']
    hps.Float('c', 0, 1, 0.1, default=0)
    assert 'c' in oracle.populate_space('1_1')['values']
    hps.Fixed('d', 7)
    assert 'd' in oracle.populate_space('1_2')['values']
    hps.Choice('e', [9, 0], default=9)
    assert 'e' in oracle.populate_space('1_3')['values']


def test_bayesian_save_reload(tmp_dir):
    hps = hp_module.HyperParameters()
    hps.Choice('a', [1, 2], default=1)
    hps.Choice('b', [3, 4], default=3)
    hps.Choice('c', [5, 6], default=5)
    hps.Choice('d', [7, 8], default=7)
    hps.Choice('e', [9, 0], default=9)
    oracle = bo_module.BayesianOptimizationOracle(
        objective='score', max_trials=20, hyperparameters=hps)

    for _ in range(3):
        trial = oracle.create_trial('tuner_id')
        oracle.update_trial(trial.trial_id, {'score': 1.})
        oracle.end_trial(trial.trial_id, "COMPLETED")

    fname = os.path.join(tmp_dir, 'oracle')
    oracle.save(fname)
    oracle = bo_module.BayesianOptimizationOracle(
        objective='score', max_trials=20, hyperparameters=hps)
    oracle.reload(fname)

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
    )
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
    oracle.populate_space(str(1))
    oracle.save(os.path.join(tmp_dir, 'temp_oracle'))
