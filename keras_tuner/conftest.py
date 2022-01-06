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

import numpy as np
import pytest
import tensorflow as tf


@pytest.fixture(autouse=True)
def set_seeds_before_tests():
    """Test wrapper to set the seed before each test.

    This wrapper runs for all the tests in the test suite.
    """
    random.seed(0)
    np.random.seed(0)
    tf.random.set_seed(0)
    yield
