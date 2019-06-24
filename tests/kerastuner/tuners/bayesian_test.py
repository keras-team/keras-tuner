import pytest
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
        model.add(tf.keras.layers.Dense(units=hp.Range('units_' + str(i),
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
    hp_list = [hp_module.Choice('a', [1, 2], default=1),
               hp_module.Range('b', 3, 10, default=3),
               hp_module.Linear('c', 0, 1, 0.1, default=0),
               hp_module.Fixed('d', 7),
               hp_module.Choice('e', [9, 0], default=9)]
    oracle = bo_module.BayesianOptimizationOracle()
    for i in range(100):
        oracle.populate_space(str(i), hp_list)
        oracle.result(str(i), i)


def test_bayesian_dynamic_space(tmp_dir):
    hp_list = [hp_module.Choice('a', [1, 2], default=1)]
    oracle = bo_module.BayesianOptimizationOracle()
    for i in range(10):
        oracle.populate_space(str(i), hp_list)
        oracle.result(str(i), i)
    hp_list.append(hp_module.Range('b', 3, 10, default=3))
    assert 'b' in oracle.populate_space('0', hp_list)['values']
    hp_list.append(hp_module.Linear('c', 0, 1, 0.1, default=0))
    assert 'c' in oracle.populate_space('1', hp_list)['values']
    hp_list.append(hp_module.Fixed('d', 7))
    assert 'd' in oracle.populate_space('2', hp_list)['values']
    hp_list.append(hp_module.Choice('e', [9, 0], default=9))
    assert 'e' in oracle.populate_space('3', hp_list)['values']


def test_bayesian_optimization_tuner(tmp_dir):
    tuner = bo_module.BayesianOptimization(
        build_model,
        objective='val_accuracy',
        max_trials=15,
    )
    assert isinstance(tuner.oracle, bo_module.BayesianOptimizationOracle)
