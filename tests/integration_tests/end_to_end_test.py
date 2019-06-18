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

"Keras Tuner hello world sequential API - TensorFlow V1.13+ or V2.x"
import numpy as np
import os
import pytest
import tempfile
from tensorflow.keras.models import Sequential, Model  # pylint: disable=import-error
from tensorflow.keras.layers import Input, Dense  # pylint: disable=import-error
from tensorflow.keras.optimizers import Adam  # pylint: disable=import-error

# Function used to specify hyper-parameters.
from kerastuner.distributions import Range, Choice, Boolean, Fixed

from kerastuner.engine.instance import Instance


# Tuner used to search for the best model. Changing hypertuning algorithm
# simply requires to change the tuner class used.
from kerastuner.tuners import RandomSearch

from kerastuner.collections.instancestatescollection import InstanceStatesCollection  # nopep8

# Defining what to hypertune is as easy as writing a Keras/TensorFlow model
# The only differences are:
# 1. Wrapping the model in a function
# 2. Defining hyperparameters as variable using distribution functions
# 3. Replacing the fixed values with the variables holding the hyperparameters
#    ranges.


def model_fn():
    "Model with hyper-parameters"

    # Hyper-parameters are defined as normal python variables
    DIMS = Range('dims', 2, 4, 2, group='layers')
    ACTIVATION = Choice('activation', ['relu', 'tanh'], group="layers")
    EXTRA_LAYER = Boolean('extra_layer', group="layers")
    LR = Choice('lr', [0.01, 0.001, 0.0001], group="optimizer")

    # converting a model to a tunable model is a simple as replacing static
    # values with the hyper parameters variables.
    model = Sequential()
    model.add(Dense(DIMS, input_shape=(1, )))
    model.add(Dense(DIMS, activation=ACTIVATION))
    if EXTRA_LAYER:
        model.add(Dense(DIMS, activation=ACTIVATION))
    model.add(Dense(1, activation='sigmoid'))
    optimizer = Adam(LR)
    model.compile(optimizer=optimizer,
                  loss="binary_crossentropy",
                  metrics=['acc'])
    return model


def fixed_model_fn():
    "Model with hyper-parameters"

    # Hyper-parameters are defined as normal python variables
    i = Input(shape=(2, ), dtype='float32')
    model = Model(i, i)
    model.compile(optimizer="adam",
                  loss="binary_crossentropy",
                  metrics=['acc'])
    return model


@pytest.fixture(scope="module")
def model_tmp_path(tmpdir_factory):
    return tmpdir_factory.mktemp('end_to_end_test')


@pytest.fixture(scope="module")
def fixed_model_tmp_path(tmpdir_factory):
    return tmpdir_factory.mktemp('fixed_end_to_end_test')


@pytest.fixture(scope="module")
def tuner(model_tmp_path):
    tmp_dir = str(model_tmp_path / "tmp")
    results_dir = str(model_tmp_path / "results")
    export_dir = str(model_tmp_path / "export")

    # Random data to feed the model.

    x_train = []
    y_train = []

    for _ in range(100):
        x_train.append(np.random.random(1) - .5)
        y_train.append(0)

        x_train.append(np.random.random(1) + .5)
        y_train.append(1)

    x_train = np.array(x_train)
    y_train = np.array(y_train)

    # Initialize the hypertuner by passing the model function (model_fn)
    # and specifying key search constraints: maximize val_acc (objective),
    # spend 9 epochs doing the search, spend at most 3 epoch on each model.
    tuner = RandomSearch(model_fn,
                         objective='val_acc',
                         epoch_budget=100,
                         max_epochs=10,
                         results_dir=results_dir,
                         tmp_dir=tmp_dir,
                         export_dir=export_dir)

    # display search overview
    tuner.summary()

    # You can use http://keras-tuner.appspot.com to track results on the web,
    # and get notifications. To do so, grab an API key on that site, and fill
    # it here.
    # tuner.enable_cloud(api_key=api_key)

    # Perform the model search. The search function has the same prototype than
    # keras.Model.fit(). Similarly search_generator() mirror
    # search_generator().
    tuner.search(x_train, y_train, validation_data=(x_train, y_train))

    return tuner


def fixed_result_tuner(fixed_model_tmp_path):
    tmp_dir = str(fixed_model_tmp_path / "tmp")
    results_dir = str(fixed_model_tmp_path / "results")
    export_dir = str(fixed_model_tmp_path / "export")

    # Random data to feed the model.

    x_train = []
    y_train = []

    for idx in range(100):
        if idx % 2 == 0:
            x_train.append([0, 1])
            y_train.append([0, 1])
        else:
            x_train.append([1, 0])
            y_train.append([1, 0])

    for idx in range(10):
        if idx % 2 == 0:
            x_train.append([0, 1])
            y_train.append([1, 0])
        else:
            x_train.append([1, 0])
            y_train.append([0, 1])

    x_train = np.array(x_train, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.float32)

    # Initialize the hypertuner by passing the model function (model_fn)
    # and specifying key search constraints: maximize val_acc (objective),
    # spend 9 epochs doing the search, spend at most 3 epoch on each model.
    tuner = RandomSearch(fixed_model_fn,
                         objective='val_acc',
                         epoch_budget=100,
                         max_epochs=10,
                         results_dir=results_dir,
                         tmp_dir=tmp_dir,
                         export_dir=export_dir)

    # display search overview
    tuner.summary()

    # You can use http://keras-tuner.appspot.com to track results on the web,
    # and get notifications. To do so, grab an API key on that site, and fill
    # it here.
    # tuner.enable_cloud(api_key=api_key)

    # Perform the model search. The search function has the same prototype than
    # keras.Model.fit(). Similarly search_generator() mirror
    # search_generator().
    tuner.search(x_train, y_train, validation_data=(x_train, y_train))

    return tuner


def test_end_to_end_summary(tuner):
    # Show the best models, their hyperparameters, and the resulting metrics.
    tuner.results_summary()


def test_end_to_end_export(tuner):
    # Export the top 2 models, in keras format format.
    tuner.save_best_models(num_models=2)


def test_end_to_end_get_best_models(tuner):
    instance_states, _, _ = tuner.get_best_models(num_models=1, compile=False)

    m = tuner.reload_instance(instance_states[0].idx)
    assert isinstance(m, Instance)


def test_end_to_end_classification_reports(fixed_model_tmp_path):
    # Show the best models, their hyperparameters, and the resulting metrics.

    fixed_result_tuner(fixed_model_tmp_path)

    path = str(fixed_model_tmp_path / "results")
    ic = InstanceStatesCollection()
    ic.load_from_dir(path)

    for instance in ic.to_list():
        for execution in instance.execution_states_collection.to_list():
            print(execution)
            assert execution.roc_auc_score is not None
            assert isinstance(execution.roc_auc_score, float)
            assert execution.roc_auc_score > 0

            # Classification metrics can be undefined if the model predicts a
            # single class for all inputs.
            if execution.classification_metrics:
                assert "macro avg" in execution.classification_metrics
                assert "weighted avg" in execution.classification_metrics

                print(execution.classification_metrics)
