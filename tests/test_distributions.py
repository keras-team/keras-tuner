from kerastuner import distributions
from collections import defaultdict

SAMPLE_SIZE = 10000

# Fixed
def test_fixed_correctness():
  tests = ['a', True, 1, 1.1, ['ab']]
  for test in tests:
    assert distributions.Fixed(test) == test

# Boolean
def test_bool_correctness():
  for _ in range(SAMPLE_SIZE):
    res = distributions.Boolean()
    assert res == True or res == False
  
def test_bool_randomness():
  res = defaultdict(int)
  for _ in range(SAMPLE_SIZE):
    x = distributions.Boolean()
    res[x] += 1
  prob = round(res[True] / float(SAMPLE_SIZE), 1)
  assert prob == 0.5

# Choice
def test_choice_correctness():
  res = distributions.Choice('a', 'b', 'c')
  assert res in ['a', 'b', 'c']

  res = distributions.Choice(1, 2, 3)
  assert res in [1, 2, 3]

  res = distributions.Choice(1.1, 2.2, 3.3)
  assert res in [1.1, 2.2, 3.3]

def test_choice_randomness():
  res = defaultdict(int)
  for _ in range(SAMPLE_SIZE):
    x = distributions.Choice('a', 'b', 'c')
    res[x] += 1
  prob = round(res['a'] / float(SAMPLE_SIZE), 1)
  assert prob == 0.3

# Range
def test_range_type_correctness():
  for _ in range(10):
    res = distributions.Range(1, 4)
    assert res in [1, 2, 3, 4]
    assert isinstance(res, int)

def test_range_increment_correctness():
  for _ in range(10):
    res = distributions.Range(2, 8, 2)
    assert res in [2, 4, 6, 8]
    assert isinstance(res, int)

def test_range_randomness():
  res = defaultdict(int)
  for _ in range(SAMPLE_SIZE):
    x = distributions.Range(1, 100)
    res[x] += 1
  prob = round(res[42] / float(SAMPLE_SIZE), 2)
  assert prob == 0.01

# Linear
def test_linear_int_correctness():
  for _ in range(100):
    res = distributions.Linear(1, 50, 6)
    assert res in [1, 10, 20, 30, 40, 50]
    assert isinstance(res, int)

def test_linear_float_correctness():
  for _ in range(100):
    res = distributions.Linear(1.0, 2.0, 11, precision=2)
    assert res in [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]
    assert isinstance(res, float)

def test_linear_int_randomness():
  res = defaultdict(int)
  num_buckets = 100
  for _ in range(SAMPLE_SIZE):
    x = distributions.Linear(1, 100, num_buckets)
    res[x] += 1
  prob = round(res[42] / float(SAMPLE_SIZE), 2)
  assert prob == 1 / float(num_buckets)

def test_linear_float_randomness():
  res = defaultdict(int)
  num_buckets = 100
  for _ in range(SAMPLE_SIZE):
    x = distributions.Linear(1.0, 100.0, num_buckets)
    res[x] += 1
  prob = round(res[42] / float(SAMPLE_SIZE), 2)
  assert prob == 1 / float(num_buckets)