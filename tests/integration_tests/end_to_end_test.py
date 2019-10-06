# Copyright 2019 The Keras Tuner Authors
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest

import tensorflow as tf
from tensorflow import keras
import kerastuner


@pytest.fixture(scope='function')
def tmp_dir(tmpdir_factory):
    return tmpdir_factory.mktemp('integration_test', numbered=True)


mnist_data = None


def get_data():
    global mnist_data
    if not mnist_data:
        mnist_data = keras.datasets.mnist.load_data()
    return mnist_data


def build_model(hp):
    inputs = keras.Input(shape=(28, 28))
    x = keras.layers.Reshape((28 * 28,))(inputs)
    for i in range(hp.Int('num_layers', 1, 4)):
        x = keras.layers.Dense(
            units=hp.Int('units_' + str(i), 128, 512, 32, default=256),
            activation='relu')(x)
    x = keras.layers.Dropout(hp.Float('dp', 0., 0.6, 0.1, default=0.5))(x)
    outputs = keras.layers.Dense(10, activation='softmax')(x)
    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(
            hp.Choice('learning_rate', [1e-2, 2e-3, 5e-4])),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'])
    return model


@pytest.mark.parametrize(
    'distribution_strategy',
    [None, tf.distribute.OneDeviceStrategy('/cpu:0')])
def test_end_to_end_workflow(tmp_dir, distribution_strategy):
    (x, y), (val_x, val_y) = get_data()
    x = x.astype('float32') / 255.
    val_x = val_x.astype('float32') / 255.

    x = x[:10000]
    y = y[:10000]

    tuner = kerastuner.tuners.RandomSearch(
        build_model,
        objective='val_accuracy',
        max_trials=20,
        distribution_strategy=distribution_strategy,
        directory=tmp_dir)

    tuner.search_space_summary()

    tuner.search(x=x,
                 y=y,
                 epochs=10,
                 batch_size=128,
                 callbacks=[keras.callbacks.EarlyStopping(patience=2)],
                 validation_data=(val_x, val_y))

    tuner.results_summary()

    best_model = tuner.get_best_models(1)[0]

    val_loss, val_acc = best_model.evaluate(val_x, val_y)
    assert val_acc > 0.955


if __name__ == '__main__':
    test_end_to_end_workflow('test_dir', None)
