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

import keras_core

from keras_tuner.backend.config import multi_backend

if multi_backend():
    from keras_core.src.ops import *  # noqa: F403, F401
else:
    from keras_core.src.backend.tensorflow import *  # noqa: F403, F401
    from keras_core.src.backend.tensorflow.core import *  # noqa: F403, F401
    from keras_core.src.backend.tensorflow.math import *  # noqa: F403, F401
    from keras_core.src.backend.tensorflow.nn import *  # noqa: F403, F401
    from keras_core.src.backend.tensorflow.numpy import *  # noqa: F403, F401


if keras_core.config.backend() == "tensorflow" or not multi_backend():

    def take_along_axis(x, indices, axis=None):
        import tensorflow as tf

        # TODO: move this workaround for dynamic shapes into keras-core.
        if axis < 0:
            axis = axis + indices.shape.rank
        # If all shapes after axis are 1, squeeze them off and use tf.gather.
        # tf.gather plays nicer with dynamic shapes in compiled functions.
        leftover_axes = list(range(axis + 1, indices.shape.rank))
        static_shape = indices.shape.as_list()
        squeezable = True
        for i in leftover_axes:
            if static_shape[i] != 1:
                squeezable = False
        if squeezable:
            if leftover_axes:
                indices = tf.squeeze(indices, leftover_axes)
            return tf.gather(x, indices, batch_dims=axis)
        # Otherwise, fall back to the tfnp call.
        return keras_core.src.backend.tensorflow.numpy.take_along_axis(
            x, indices, axis=axis
        )
