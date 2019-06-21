import tensorflow as tf
from kerastuner.engine import hyperparameters as hp_module
from kerastuner.tuners import bayesian as bo_module


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


def test_bayesian_oracle():
    hp_list = [hp_module.Choice('a', [1, 2], default=1),
               hp_module.Range('b', 3, 10, default=3),
               hp_module.Linear('c', 0, 1, 0.1, default=0),
               hp_module.Fixed('d', 7),
               hp_module.Choice('e', [9, 0], default=9)]
    oracle = bo_module.BayesianOptimizationOracle()
    for i in range(10):
        oracle.populate_space(str(i), hp_list)
        oracle.result(str(i), i)


def test_bayesian_optimization_tuner():
    tuner = bo_module.BayesianOptimization(
        build_model,
        objective='val_accuracy',
        max_trials=15,
    )
    assert isinstance(tuner.oracle, bo_module.BayesianOptimizationOracle)

