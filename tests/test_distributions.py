from kerastuner import distributions
from collections import defaultdict
import pytest

SAMPLE_SIZE = 10000


@pytest.fixture
def clean_hyper():
    distributions.hyper_parameters = {}

#hyper parameters recording
def test_single_param(clean_hyper):
  x = distributions.Boolean("test")
  assert "default:test" in distributions.hyper_parameters

def test_duplicate_param_name_diff_group(clean_hyper):

  x = distributions.Boolean("test", group='a')
  x = distributions.Boolean("test", group='b')
  assert len(distributions.hyper_parameters) == 2

def test_duplicate_param_name_same_group(clean_hyper):
  x = distributions.Boolean("test", group='a')
  with pytest.raises(ValueError):
    x = distributions.Boolean("test", group='a')

# Fixed
def test_fixed_correctness(clean_hyper):
  tests = ['a', True, 1, 1.1, ['ab']]
  for test in tests:
    assert distributions.Fixed("test", test) == test

# Boolean
def test_bool_correctness(clean_hyper):
  for _ in range(SAMPLE_SIZE):
    res = distributions.Boolean("test")
    assert res == True or res == False
  
def test_bool_randomness(clean_hyper):
  res = defaultdict(int)
  for _ in range(SAMPLE_SIZE):
    x = distributions.Boolean("test")
    res[x] += 1
  prob = round(res[True] / float(SAMPLE_SIZE), 1)
  assert prob == 0.5

# Choice
def test_choice_correctness(clean_hyper):
  choices = [
    [['a', 'b', 'c'], str],
    [[1, 2, 3], int],
    [[1.1, 2.2, 3.3], float]
  ]
  for choice in choices:
    res = distributions.Choice("test", choice[0])
    assert res in choice[0]
    assert isinstance(res, choice[1])

def test_choice_randomness(clean_hyper):
  res = defaultdict(int)
  for _ in range(SAMPLE_SIZE):
    x = distributions.Choice("test", ['a', 'b', 'c'])
    res[x] += 1
  prob = round(res['a'] / float(SAMPLE_SIZE), 1)
  assert prob == 0.3

# Range
def test_range_type_correctness(clean_hyper):
  for _ in range(10):
    res = distributions.Range('test', 1, 4)
    assert res in [1, 2, 3, 4]
    assert isinstance(res, int)

def test_range_increment_correctness(clean_hyper):
  for _ in range(10):
    res = distributions.Range("test", 2, 8, 2)
    assert res in [2, 4, 6, 8]
    assert isinstance(res, int)

def test_range_randomness(clean_hyper):
  res = defaultdict(int)
  for _ in range(SAMPLE_SIZE):
    x = distributions.Range("test", 1, 100)
    res[x] += 1
  prob = round(res[42] / float(SAMPLE_SIZE), 2)
  assert prob == 0.01

# Linear
def test_linear_int_correctness(clean_hyper):
  for _ in range(100):
    res = distributions.Linear("test", 1, 50, 6)
    assert res in [1, 10, 20, 30, 40, 50]
    assert isinstance(res, int)

def test_linear_float_correctness(clean_hyper):
  for _ in range(100):
    res = distributions.Linear("test", 1.0, 2.0, 11, precision=2)
    assert res in [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]
    assert isinstance(res, float)

def test_linear_int_randomness(clean_hyper):
  res = defaultdict(int)
  num_buckets = 100
  for _ in range(SAMPLE_SIZE):
    x = distributions.Linear("test", 1, 100, num_buckets)
    res[x] += 1
  prob = round(res[42] / float(SAMPLE_SIZE), 2)
  assert prob == 1 / float(num_buckets)

def test_linear_float_randomness(clean_hyper):
  res = defaultdict(int)
  num_buckets = 100
  for _ in range(SAMPLE_SIZE):
    x = distributions.Linear("test", 1.0, 100.0, num_buckets)
    res[x] += 1
  prob = round(res[42] / float(SAMPLE_SIZE), 2)
  assert prob == 1 / float(num_buckets)