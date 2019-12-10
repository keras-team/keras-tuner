import pytest

import numpy as np

from tensorflow import keras
import kerastuner
from mock import patch


class OkResponse(object):
    ok = True


class MockPost(object):

    def __init__(self):
        self.url_calls = []

    def __call__(self, url, headers, json):
        self.url_calls.append(url)
        return OkResponse()


MOCK_POST = MockPost()

INPUT_DIM = 2
NUM_CLASSES = 3
NUM_SAMPLES = 64
TRAIN_INPUTS = np.random.random(size=(NUM_SAMPLES, INPUT_DIM))
TRAIN_TARGETS = np.random.randint(0, NUM_CLASSES, size=(NUM_SAMPLES, 1))
VAL_INPUTS = np.random.random(size=(NUM_SAMPLES, INPUT_DIM))
VAL_TARGETS = np.random.randint(0, NUM_CLASSES, size=(NUM_SAMPLES, 1))


@pytest.fixture(scope='function')
def tmp_dir(tmpdir_factory):
    return tmpdir_factory.mktemp('integration_test', numbered=True)


def build_model(hp):
    inputs = keras.Input(shape=(INPUT_DIM,))
    x = inputs
    for i in range(hp.Int('num_layers', 1, 4)):
        x = keras.layers.Dense(
            units=hp.Int('units_' + str(i), 5, 9, 1, default=6),
            activation='relu')(x)
    outputs = keras.layers.Dense(NUM_CLASSES, activation='softmax')(x)
    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(
            hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4])),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'])
    return model


@patch('requests.post', MOCK_POST)
def test_cloud_logger(tmp_dir):
    cloud_logger = kerastuner.engine.logger.CloudLogger('123')

    tuner = kerastuner.tuners.RandomSearch(
        build_model,
        objective='val_accuracy',
        max_trials=6,
        directory=tmp_dir,
        logger=cloud_logger)

    assert tuner.logger is cloud_logger

    tuner.search(x=TRAIN_INPUTS,
                 y=TRAIN_TARGETS,
                 epochs=2,
                 validation_data=(VAL_INPUTS, VAL_TARGETS))
    assert len(MOCK_POST.url_calls) >= 10
    keys = [
        ('register_tuner', 1),
        ('register_trial', 6),
    ]
    for key, count in keys:
        actual_count = 0
        for url in MOCK_POST.url_calls:
            if key in url:
                actual_count += 1
        assert count == actual_count
