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

import numpy as np
import tensorflow as tf

from kerastuner.tuners.ultraband import UltraBand


def build_model(hp):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=(2, 2)))
    for i in range(3):
        model.add(tf.keras.layers.Dense(units=hp.Range('units_' + str(i), 2, 4, 2),
                                        activation='relu'))
    model.add(tf.keras.layers.Dense(2, activation='softmax'))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4])),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'])
    return model


def test_ultraband_tuner():
    x = np.random.rand(10, 2, 2).astype('float32')
    y = np.random.randint(0, 1, (10, ))
    val_x = np.random.rand(10, 2, 2).astype('float32')
    val_y = np.random.randint(0, 1, (10, ))

    tuner = UltraBand(
        build_model,
        objective='val_accuracy',
        max_trials=15,
        executions_per_trial=3,
        directory='test_dir')

    tuner.search_space_summary()

    tuner.search(x=x,
                 y=y,
                 epochs=1,
                 validation_data=(val_x, val_y))

    tuner.results_summary()

