import pytest

from kerastuner.distributions import Fixed, Boolean, Choice, Range
from kerastuner.distributions import Linear, Logarithmic, reset_distributions


def test_clear_distributions():
    Fixed('test_dup_succeed', 3)
    reset_distributions()
    Fixed('test_dup_succeed', 3)


def test_duplicate_error():
    "by default DISTRIBUTIONS should the dummydistribution and catch the error"
    Fixed('test_dup_error', 3)
    with pytest.raises(ValueError):
        Fixed('test_dup_error', 3)


def test_fixed():
    assert Fixed('test_fixed', 3) == 3


def test_bool():
    assert Boolean('test_bool')


def test_choice():
    assert Choice('test_choice', [3, 2, 1]) == 3


def test_choice_invalid_type():
    with pytest.raises(ValueError):
        Choice('test_choice_invalid_type', 3)


# Range
def test_range():
    assert Range('test_range', 7, 20, 5) == 7


def test_range_invalid_start():
    with pytest.raises(ValueError):
        Range('test_range_invalid_start', 'a', 3, 4)


def test_range_invalid_stop():
    with pytest.raises(ValueError):
        Range('test_range_invalid_stop', 1, 'a', 4)


def test_range_invalid_increment():
    with pytest.raises(ValueError):
        Range('test_range_invalid_increment', 1, 3, 'a')


def test_range_stop_larger_than_start():
    with pytest.raises(ValueError):
        Range('test_range_stop_larger_than_start', 3, 1)


def test_range_incr_larger_than_range():
    with pytest.raises(ValueError):
        Range('test_range_incr_larger_than_range', 1, 3, 10)


# linear
def test_linear():
    assert Linear('test_linear', 7, 20, 19) == 7


def test_linear_invalid_start():
    with pytest.raises(ValueError):
        Linear('test_linear_invalid_start', 'a', 3, 4)


def test_linear_invalid_stop():
    with pytest.raises(ValueError):
        Linear('test_linear_invalid_stop', 1, 'a', 4)


def test_linear_invalid_bucket():
    with pytest.raises(ValueError):
        Linear('test_linear_invalid_bucket', 1, 3, 'a')


def test_linear_stop_larger_than_start():
    with pytest.raises(ValueError):
        Linear('test_linear_stop_larger_than_start', 3, 1, 3)


# Logarithmic
def test_logarithmic():
    assert Logarithmic('test_logarithmic', 7, 20, 19) == 7


def test_logarithmic_invalid_start():
    with pytest.raises(ValueError):
        Logarithmic('test_logarithmic_invalid_start', 'a', 3, 4)


def test_logarithmic_invalid_stop():
    with pytest.raises(ValueError):
        Logarithmic('test_logarithmic_invalid_stop', 1, 'a', 4)


def test_logarithmic_invalid_bucket():
    with pytest.raises(ValueError):
        Logarithmic('test_logarithmic_invalid_bucket', 1, 3, 'a')


def test_logarithmic_stop_larger_than_start():
    with pytest.raises(ValueError):
        Logarithmic('test_logarithmic_stop_larger_than_start', 3, 1, 3)
