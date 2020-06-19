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
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist

from kerastuner import RandomSearch

TRIALS = 3  # number of models to train
EPOCHS = 2  # number of epoch per model

# Get the MNIST dataset.
(x_train, y_train), (x_val, y_val) = mnist.load_data()
x_train = np.expand_dims(x_train.astype('float32') / 255, -1)
x_val = np.expand_dims(x_val.astype('float32') / 255, -1)
y_train = to_categorical(y_train, 10)
y_val = to_categorical(y_val, 10)


def build_model(hp):
    """Function that build a TF model based on hyperparameters values.

    Args:
        hp (HyperParameter): hyperparameters values

    Returns:
        Model: Compiled model
    """

    num_layers = hp.Int('num_layers', 2, 8, default=6)
    lr = hp.Choice('learning_rate', [1e-3, 5e-4])

    inputs = layers.Input(shape=(28, 28, 1))
    x = inputs

    for idx in range(num_layers):
        idx = str(idx)

        filters = hp.Int('filters_' + idx, 32, 256, step=32, default=64)
        x = layers.Conv2D(filters=filters, kernel_size=3, padding='same',
                          activation='relu')(x)

        # add a pooling layers if needed
        if x.shape[1] >= 8:
            pool_type = hp.Choice('pool_' + idx, values=['max', 'avg'])
            if pool_type == 'max':
                x = layers.MaxPooling2D(2)(x)
            elif pool_type == 'avg':
                x = layers.AveragePooling2D(2)(x)

    x = layers.Flatten()(x)
    outputs = layers.Dense(10, activation='softmax')(x)

    # Build model
    model = keras.Model(inputs, outputs)
    model.compile(optimizer=Adam(lr),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


# Initialize the tuner by passing the `build_model` function
# and specifying key search constraints: maximize val_acc (objective),
# and the number of trials to do. More efficient tuners like UltraBand() can
# be used.
tuner = RandomSearch(build_model, objective='val_accuracy', max_trials=TRIALS,
                     project_name='hello_world_tutorial_results')

# Display search space overview
tuner.search_space_summary()

# Perform the model search. The search function has the same signature
# as `model.fit()`.
tuner.search(x_train, y_train, batch_size=128, epochs=EPOCHS,
             validation_data=(x_val, y_val))

# Display the best models, their hyperparameters, and the resulting metrics.
tuner.results_summary()

# Retrieve the best model and display its architecture
best_model = tuner.get_best_models(num_models=1)[0]
best_model.summary()
