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

import random
import sys

import numpy as np
import tensorflow as tf


def test_seed_set_before_each_test():
    """Test that the tests are deterministic, locally and in the CI."""
    assert random.random() == 0.8444218515250481
    assert np.random.uniform() == 0.5488135039273248

    random_tf_value = tf.random.uniform([1], dtype=tf.dtypes.float64)[0].numpy()
    if sys.version_info >= (3, 0):
        assert random_tf_value == 0.3358018290373803
    else:
        assert random_tf_value == 0.3759982263246542
