import pytest

from kerastuner.engine.tuner import Tuner
from kerastuner.distributions.randomdistributions import RandomDistributions


def test_empty_model_function():
    with pytest.raises(ValueError):
        Tuner(None, 'test', 'loss', RandomDistributions)

# FIXME: test invalid model function