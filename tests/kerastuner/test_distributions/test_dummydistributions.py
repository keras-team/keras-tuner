import pytest

from kerastuner.distributions import DummyDistributions
from .common import fixed_correctness_test, bool_correctness_test
from .common import choice_correctness_test, range_type_correctness_test
from .common import linear_correctness_test, logarithmic_correctness_test


# hyper parameters
def check_params(hp_config, distributions, name, group, hp_type,
                 space_size, start, stop):
    key = distributions._get_key(name, group)
    val = hp_config[key]
    assert val['name'] == name
    assert val['group'] == group
    assert val['type'] == hp_type
    assert val['space_size'] == space_size
    assert val['start'] == start
    assert val['stop'] == stop


def test_boolean_param_recording():
    name = "myname"
    group = "mygroup"
    distributions = DummyDistributions()
    distributions.Boolean("myname", group='mygroup')
    hp_config = distributions.get_hyperparameters_config()
    check_params(hp_config, distributions, name, group, 'Boolean',
                 2, True, False)


def test_fixed_param_recording():
    name = "myname"
    group = "mygroup"
    distributions = DummyDistributions()
    distributions.Fixed("myname", 42, group='mygroup')
    hp_config = distributions.get_hyperparameters_config()
    check_params(hp_config, distributions, name, group, 'Fixed',
                 1, 42, 42)


def test_choice_param_recording():
    name = "myname"
    group = "mygroup"
    distributions = DummyDistributions()
    distributions.Choice("myname", ['a', 'b', 'c'], group='mygroup')
    hp_config = distributions.get_hyperparameters_config()
    check_params(hp_config, distributions, name, group, 'Choice',
                 3, 'a', 'c')


def test_range_param_recording():
    name = "myname"
    group = "mygroup"
    distributions = DummyDistributions()
    distributions.Range("myname", 1, 10, 2, group='mygroup')
    hp_config = distributions.get_hyperparameters_config()
    check_params(hp_config, distributions, name, group, 'Range',
                 5, 1, 10)


def test_linear_param_recording():
    name = "myname"
    group = "mygroup"
    distributions = DummyDistributions()
    distributions.Linear("myname", 1, 10, 10, group='mygroup')
    hp_config = distributions.get_hyperparameters_config()
    check_params(hp_config, distributions, name, group, 'Linear',
                 10, 1, 10)


def test_logarithmic_param_recording():
    name = "myname"
    group = "mygroup"
    distributions = DummyDistributions()
    distributions.Logarithmic("myname", 1, 10, 10, group='mygroup')
    hp_config = distributions.get_hyperparameters_config()
    check_params(hp_config, distributions, name, group, 'Logarithmic',
                 10, 1, 10)


def test_duplicate_param_name_diff_group():
    distributions = DummyDistributions()
    distributions.Boolean("test", group='a')
    distributions.Boolean("test", group='b')
    hp_config = distributions.get_hyperparameters_config()
    assert len(hp_config) == 2


def test_duplicate_param_name_same_group():
    distributions = DummyDistributions()
    distributions.Boolean("test", group='a')
    with pytest.raises(ValueError):
        distributions.Boolean("test", group='a')


# Fixed
def test_fixed_correctness():
    fixed_correctness_test(DummyDistributions())


# Boolean
def test_bool_correctness():
    bool_correctness_test(DummyDistributions())


# Choice
def choice_range_correctness():
    choice_correctness_test(DummyDistributions())


# Range
def range_range_correctness():
    range_type_correctness_test(DummyDistributions())


# Linear
def test_linear_correctness():
    linear_correctness_test(DummyDistributions())


# Logarithmic
def test_logarithmic_correctness():
    res = DummyDistributions().Logarithmic('test', 1, 2, 10)
    assert res == 1
