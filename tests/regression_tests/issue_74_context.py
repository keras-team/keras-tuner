import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers


def build_model(hp):
    """Model function based on:
    https://github.com/keras-team/keras-tuner/issues/74
    """
    model = keras.Sequential()
    for i in range(hp.Int('num_layers', 2, 20)):
        model.add(
            layers.Dense(units=hp.Int('units_' + str(i),
                                      min_value=32,
                                      max_value=512,
                                      step=32),
                         activation='relu'))
    model.add(layers.Dense(24, activation='sigmoid'))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer=keras.optimizers.Adam(
        hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4])),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


def _make_dataset():
    """Make a meaningful but simple dataset where there are 24 classes,
    and the x values for each class range from [0-CLASS_NUM), suitable for use
    with the model from issue 74.   
    """    
    x = []
    y = []
    for idx in range(1024):
        x.append(np.random.rand(128) * (idx % 24))
        y.append(idx % 24)
    x = np.array(x)
    y = np.array(y)

    return x, y


def make_testdata():
    """Make a training data / test data pair, suitable for use with the
    model from issue 74"""
    x_train, y_train = _make_dataset()
    x_test, y_test = _make_dataset()

    return (x_train, y_train), (x_test, y_test)
