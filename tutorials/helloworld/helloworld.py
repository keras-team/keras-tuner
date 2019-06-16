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
"""Keras Tuner hello world sequential API - TensorFlow V1.13+ or V2.x."""

import argparse
import os
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping  # pylint: disable=import-error
from tensorflow.keras.models import Sequential  # pylint: disable=import-error
from tensorflow.keras.layers import Dense  # pylint: disable=import-error
from tensorflow.keras.optimizers import Adam  # pylint: disable=import-error

# Function used to specify hyper-parameters.
from kerastuner.distributions import Range, Choice, Boolean, Fixed

# Tuner used to search for the best model. Changing hypertuning algorithm
# simply requires to change the tuner class used.
from kerastuner.tuners import RandomSearch
from kerastuner.tuners import UltraBand

# Random data to feed the model.
x_train = np.random.random((500, 200))
y_train = np.random.randint(2, size=(500, 1))


# Defining what to hypertune is as easy as writing a Keras/TensorFlow model
# The only differences are:
# 1. Wrapping the model in a function
# 2. Defining hyperparameters as variable using distribution functions
# 3. Replacing the fixed values with the variables holding the hyperparameters
#    ranges.
def model_fn():
    "Model with hyper-parameters"

    # Hyper-parameters are defined as normal python variables
    DIMS = Range('dims', 16, 32, 2, group='layers')
    ACTIVATION = Choice('activation', ['relu', 'tanh'], group="layers")
    EXTRA_LAYER = Boolean('extra_layer', group="layers")
    LR = Choice('lr', [0.01, 0.001, 0.0001], group="optimizer")

    # converting a model to a tunable model is a simple as replacing static
    # values with the hyper parameters variables.
    model = Sequential()
    model.add(Dense(DIMS, input_shape=(200, )))
    model.add(Dense(DIMS, activation=ACTIVATION))
    if EXTRA_LAYER:
        model.add(Dense(DIMS, activation=ACTIVATION))
    model.add(Dense(1, activation='sigmoid'))
    optimizer = Adam(LR)
    model.compile(optimizer=optimizer,
                  loss="binary_crossentropy",
                  metrics=['accuracy'])
    return model

def main(args):
    # Suppress INFO logs coming from th tensorflow implementation, to make
    # information about the search easier to read.
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    # Initialize the hypertuner by passing the model function (model_fn)
    # and specifying key search constraints: maximize val_acc (objective),
    # spend 9 epochs doing the search, spend at most 3 epoch on each model.
    tuner = UltraBand(
        model_fn,
        objective='val_acc',
        epoch_budget=256,
        min_epochs=3,
        max_epochs=27,
        results_dir=os.path.join(args.output_directory, "results"),
        export_dir=os.path.join(args.output_directory, "export"))

    # display search overview
    tuner.summary()

    # You can use http://keras-tuner.appspot.com to track results on the web,
    # and get notifications. To do so, grab an API key on that site, and fill
    # it here.
    #
    # tuner.enable_cloud(api_key=api_key)

    # Perform the model search. The search function has the same prototype than
    # keras.Model.fit(). Similarly search_generator() mirror search_generator().
    tuner.search(x_train,
                 y_train,
                 validation_split=0.01,
                 callbacks=[EarlyStopping(monitor='val_accuracy', patience=0)])

    # Show the best models, their hyperparameters, and the resulting metrics.
    tuner.results_summary()

    # Export the top 2 models, in keras format format.
    tuner.save_best_models(num_models=2)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Showcases the Keras Tuner API.")

    parser.add_argument(
        "--output_directory",
        "-o",
        type=str,
        default=".",
        help="Path where the models and results from the tuner will be stored")

    return parser.parse_args()


if __name__ == '__main__':
    main(parse_args())
