import pytest

from tensorflow.keras.layers import Input, Dense  # nopep8 pylint: disable=import-error
from tensorflow.keras.models import Model  # pylint: disable=import-error

from kerastuner.engine.tuner import Tuner
from kerastuner.distributions.randomdistributions import RandomDistributions


def test_empty_model_function():
    with pytest.raises(ValueError):
        Tuner(None, 'test', 'loss', RandomDistributions)


def dummy():
    return 1


def test_invalid_model_function():
    with pytest.raises(ValueError):
        Tuner(dummy, 'test', 'loss', RandomDistributions)


def basic_model():
    # *can't pass as fixture as the tuner expect to call it itself
    i = Input(shape=(1,), name="input")
    o = Dense(4, name="output")(i)
    model = Model(inputs=i, outputs=o)
    model.compile(optimizer="adam", loss='mse', metrics=['accuracy'])
    return model


def test_valid_model_function():
    Tuner(basic_model, 'test', 'loss', RandomDistributions)
