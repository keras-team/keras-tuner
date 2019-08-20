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

"""Keras Tuner hello world with MNIST."""

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from kerastuner import Hyperband

# Get the MNIST dataset.
(x_train, y_train), (x_val, y_val) = keras.datasets.mnist.load_data()
x_train = x_train.astype('float32') / 255.
x_train = np.expand_dims(x_train, -1)
x_val = x_val.astype('float32') / 255.
x_val = np.expand_dims(x_val, -1)

# Define a function that returns a compiled model.
# The functions gets a `hp` argument which is used
# to query hyperparameters, such as
# `hp.Int('num_layers', 2, 8)`.


def build_model(hp):
    inputs = keras.Input(shape=(28, 28, 1))
    x = inputs
    for i in range(hp.Int('num_layers', 2, 8, default=6)):
        x = layers.Conv2D(
            filters=hp.Int('units_' + str(i), 32, 256,
                           step=32,
                           default=64),
            kernel_size=3,
            activation='relu',
            padding='same',
        )(x)
        pool = hp.Choice('pool_' + str(i),
                         values=[None, 'max', 'avg'],
                         default='max')
        if x.shape[1] >= 8:
            if pool == 'max':
                x = layers.MaxPooling2D(2)(x)
            elif pool == 'avg':
                x = layers.AveragePooling2D(2)(x)

    x = layers.Flatten()(x)
    outputs = layers.Dense(10, activation='softmax')(x)

    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=hp.Choice('learning_rate', [1e-3, 5e-4])
        ),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# Initialize the tuner by passing the `build_model` function
# and specifying key search constraints: maximize val_acc (objective),
# and spend 40 trials doing the search.


tuner = Hyperband(
    build_model,
    objective='val_accuracy',
    max_trials=40,
    min_epochs=3,
    max_epochs=20,
    directory='test_directory')

# Display search space overview
tuner.search_space_summary()

# You can use http://keras-tuner.appspot.com to track results on the web, and
# get notifications. To do so, grab an API key on that site, and fill it here.
# tuner.enable_cloud(api_key=api_key)

# Perform the model search. The search function has
# the same signature as `model.fit()`.
tuner.search(x_train, y_train,
             batch_size=128,
             validation_data=(x_val, y_val),
             callbacks=[
                keras.callbacks.EarlyStopping(
                    monitor='val_accuracy', patience=1)])

# Show the best models, their hyperparameters, and the resulting metrics.
tuner.results_summary()

# Retrieve the best model.
best_model = tuner.get_best_models(num_models=1)[0]

# Evaluate the best model.
loss, accuracy = best_model.evaluate(x_val, y_val)
