# Copyright 2019 The KerasTuner Authors
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

from unittest.mock import patch

import numpy as np
from tensorflow import keras

import keras_tuner


class OkResponse:
    ok = True


class MockPost:
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


def build_model(hp):
    inputs = keras.Input(shape=(INPUT_DIM,))
    x = inputs
    for i in range(hp.Int("num_layers", 1, 4)):
        x = keras.layers.Dense(
            units=hp.Int(f"units_{str(i)}", 5, 9, 1, default=6), activation="relu"
        )(x)

    outputs = keras.layers.Dense(NUM_CLASSES, activation="softmax")(x)
    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(
            hp.Choice("learning_rate", [1e-2, 1e-3, 1e-4])
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


@patch("requests.post", MOCK_POST)
def test_cloud_logger(tmp_path):
    cloud_logger = keras_tuner.engine.logger.CloudLogger("123")

    tuner = keras_tuner.tuners.RandomSearch(
        build_model,
        objective="val_accuracy",
        max_trials=6,
        directory=tmp_path,
        logger=cloud_logger,
    )

    assert tuner.logger is cloud_logger

    tuner.search(
        x=TRAIN_INPUTS,
        y=TRAIN_TARGETS,
        epochs=2,
        validation_data=(VAL_INPUTS, VAL_TARGETS),
    )
    assert len(MOCK_POST.url_calls) >= 10
    keys = [
        ("register_tuner", 1),
        ("register_trial", 6),
    ]
    for key, count in keys:
        actual_count = sum(key in url for url in MOCK_POST.url_calls)
        assert count == actual_count
