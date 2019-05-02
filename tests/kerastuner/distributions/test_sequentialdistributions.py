from collections import defaultdict
import pytest

from kerastuner.distributions import SequentialDistributions
from .common import record_hyperparameters_test, json_serialize_test
from .common import fixed_correctness_test, bool_correctness_test
from .common import choice_correctness_test, range_type_correctness_test
from .common import linear_correctness_test, logarithmic_correctness_test


@pytest.fixture()
def seq():
    return SequentialDistributions({})

# Hyperparameters
def test_record_hyperparameters():
    record_hyperparameters_test(SequentialDistributions({}))


# check proper use of name and group
def test_naming(seq):
    assert seq.Choice('choice', [1, 2, 3]) == 1
    assert seq.Choice('choice', [1, 2, 3], group='test2') == 1
    assert seq.Choice('choice', [1, 2, 3]) == 2


# Fixed
def test_fixed_correctness():
    fixed_correctness_test(SequentialDistributions({}))


def test_fixed_serialize(seq):
    json_serialize_test(seq.Fixed('rand', 1))


# Boolean
def test_bool_correctness(seq):
    bool_correctness_test(SequentialDistributions({}))


def test_bool_serialize(seq):
    json_serialize_test(seq.Boolean('bool'))


def test_bool_sequential(seq):
    assert seq.Boolean('bool')
    assert not seq.Boolean('bool')
    assert seq.Boolean('bool')


# Choice
def test_choice_serialize(seq):
    tests = [
        seq.Choice('choice', [1, 2, 3]),
        seq.Choice('choice', [1.0, 2.0, 3.0]),
        seq.Choice('choice', ['a', 'b', 'c']),
    ]
    for test in tests:
        json_serialize_test(test)


def test_choice_sequential(seq):
    assert seq.Choice('choice', [1, 2, 3]) == 1
    assert seq.Choice('choice', [1, 2, 3]) == 2
    assert seq.Choice('choice', [1, 2, 3]) == 3
    assert seq.Choice('choice', [1, 2, 3]) == 1
