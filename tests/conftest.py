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
