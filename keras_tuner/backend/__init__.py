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
"""
Keras backend module.

This module adds a temporarily Keras API surface that is fully under KerasTuner
control. This allows us to switch between `keras_core` and `tf.keras`, as well
as add shims to support older version of `tf.keras`.

- `config`: check which backend is being run.
- `keras`: The full `keras` API (via `keras_core` or `tf.keras`).
- `ops`: `keras_core.ops`, always tf backed if using `tf.keras`.
- `random`: `keras_core.random`, always tf backed if using `tf.keras`.
"""

from keras_tuner.backend import config
from keras_tuner.backend import io
from keras_tuner.backend import keras
from keras_tuner.backend import ops
from keras_tuner.backend import random
