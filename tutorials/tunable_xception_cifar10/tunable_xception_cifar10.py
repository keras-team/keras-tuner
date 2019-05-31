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

"""Example on how to use Tunable Resnet."""

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import cifar10
from kerastuner.applications.tunable_xception import TunableXception
from kerastuner.tuners.randomsearch import RandomSearch


# Import the Cifar10 dataset.
NUM_CLASSES = 10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
y_train = to_categorical(y_train, NUM_CLASSES)
y_test = to_categorical(y_test, NUM_CLASSES)

# Import an hypertunable version of Xception.
model_fn = TunableXception(
    input_shape=x_train.shape[1:],
    num_classes=NUM_CLASSES)

# Initialize the hypertuner: we should find the model that maximixes the
# validation accuracy, training each model for three epochs for a max of
# 12 epochs of total training time.
tuner = RandomSearch(
    model_fn,
    objective='val_acc',
    epoch_budget=12,
    max_epochs=3,
    project='Cifar10',
    architecture='Xception',
    validation_data=(x_test, y_test),
    max_params=50000000)

# Display search overview.
tuner.summary()

# Performs the hypertuning.
tuner.search(x_train, y_train, validation_split=0.01)

# Show the best models, their hyperparameters, and the resulting metrics.
tuner.results_summary()

# Export the top 2 models, in keras format format.
tuner.save_best_models(num_models=1)
