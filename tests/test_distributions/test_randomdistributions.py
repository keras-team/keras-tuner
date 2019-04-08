from collections import defaultdict

from kerastuner.distributions import RandomDistributions
from .common import record_hyperparameters_test
from .common import fixed_correctness_test, bool_correctness_test
from .common import choice_correctness_test, range_type_correctness_test
from .common import linear_correctness_test, logarithmic_correctness_test

# number of elements to draw - sample_size < 1000 cause flakiness
SAMPLE_SIZE = 10000


# Hyperparameters
def test_record_hyperparameters():
    record_hyperparameters_test(RandomDistributions())


# Fixed
def test_fixed_correctness():
    fixed_correctness_test(RandomDistributions())


# Boolean
def test_bool_correctness():
    bool_correctness_test(RandomDistributions())


def test_bool_randomness():
    distributions = RandomDistributions()
    res = defaultdict(int)
    for _ in range(SAMPLE_SIZE):
        x = distributions.Boolean("test")
        res[x] += 1
    prob = round(res[True] / float(SAMPLE_SIZE), 1)
    assert prob == 0.5


# Choice
def choice_range_correctness():
    choice_correctness_test(RandomDistributions())


def test_choice_randomness():
    distributions = RandomDistributions()
    res = defaultdict(int)
    for _ in range(SAMPLE_SIZE):
        x = distributions.Choice("test", ['a', 'b', 'c'])
        res[x] += 1
    prob = round(res['a'] / float(SAMPLE_SIZE), 1)
    assert prob == 0.3


# Range
def range_range_correctness():
    range_type_correctness_test(RandomDistributions())


def test_range_randomness():
    distributions = RandomDistributions()
    res = defaultdict(int)
    for _ in range(SAMPLE_SIZE):
        x = distributions.Range("test", 1, 100)
        res[x] += 1
    prob = round(res[42] / float(SAMPLE_SIZE), 2)
    assert prob == 0.01


# Linear
def test_linear_correctness():
    linear_correctness_test(RandomDistributions())


def test_linear_int_randomness():
    distributions = RandomDistributions()
    res = defaultdict(int)
    num_buckets = 100
    for _ in range(SAMPLE_SIZE):
        x = distributions.Linear("test", 1, 100, num_buckets)
        res[x] += 1
    prob = round(res[42] / float(SAMPLE_SIZE), 1)
    assert prob == round(1 / float(num_buckets), 1)


def test_linear_float_randomness():
    distributions = RandomDistributions()
    res = defaultdict(int)
    num_buckets = 100
    for _ in range(SAMPLE_SIZE):
        x = distributions.Linear("test", 1.0, 100.0, num_buckets)
        res[x] += 1
    prob = round(res[42] / float(SAMPLE_SIZE), 1)
    assert prob == round(1 / float(num_buckets), 1)


# Logarithmic
def test_logarithmic_correctness():
    logarithmic_correctness_test(RandomDistributions())
