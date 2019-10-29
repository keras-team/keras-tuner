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

"""Example on how to use Tunable Xception."""

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import cifar10
from kerastuner.applications import HyperXception
from kerastuner import RandomSearch

# Import the Cifar10 dataset.
NUM_CLASSES = 10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
y_train = to_categorical(y_train, NUM_CLASSES)
y_test = to_categorical(y_test, NUM_CLASSES)

# Import an hypertunable version of Xception.
hypermodel = HyperXception(
    input_shape=x_train.shape[1:],
    classes=NUM_CLASSES)

# Initialize the hypertuner: we should find the model that maximixes the
# validation accuracy, using 40 trials in total.
tuner = RandomSearch(
    hypermodel,
    objective='val_accuracy',
    max_trials=40,
    project_name='cifar10_xception',
    directory='test_directory')

# Display search overview.
tuner.search_space_summary()

# Performs the hypertuning.
tuner.search(x_train, y_train, epochs=10, validation_split=0.1)

# Show the best models, their hyperparameters, and the resulting metrics.
tuner.results_summary()

# Retrieve the best model.
best_model = tuner.get_best_models(num_models=1)[0]

# Evaluate the best model.
loss, accuracy = best_model.evaluate(x_test, y_test)
print('loss:', loss)
print('accuracy:', accuracy)
