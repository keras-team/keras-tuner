"common distributions test functions"
import json


# hyperparamters handling
def record_hyperparameters_test(distributions):
    distributions.Fixed("test", 42, group='a')
    distributions.Boolean("test", group='b')
    hp_curr = distributions.get_hyperparameters()
    assert len(hp_curr) == 2
    k = distributions._get_key('test', 'a')
    assert hp_curr[k]['name'] == 'test'
    assert hp_curr[k]['group'] == 'a'
    assert hp_curr[k]['value'] == 42


# serializable
def json_serialize_test(config):
    json.dumps(config)


# Fixed
def fixed_correctness_test(distributions):
    tests = ['a', True, 1, 1.1, ['ab']]
    for idx, test in enumerate(tests):
        assert distributions.Fixed("test", test, group=idx) == test


# Boolean
def bool_correctness_test(distributions):
    for idx in range(10):
        res = distributions.Boolean("test", group=idx)
        assert res in [True, False]


# Choice
def choice_correctness_test(distributions):
    choices = [
        [['a', 'b', 'c'], str],
        [[1, 2, 3], int],
        [[1.1, 2.2, 3.3], float]
    ]
    for idx, choice in enumerate(choices):
        res = distributions.Choice("test", choice[0], group=idx)
        assert res in choice[0]
        assert isinstance(res, choice[1])


# Range
def range_type_correctness_test(distributions):
    for idx in range(10):
        res = distributions.Range('test', 1, 4, group=idx)
        assert res in [1, 2, 3, 4]
        assert isinstance(res, int)


# Linear
def linear_correctness_test(distributions):
    for idx in range(10):
        res = distributions.Linear("test", 1.0, 2.0, 11,
                                   precision=1, group=idx)
        assert res in [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]
        assert isinstance(res, float)


# Logarithmic
def logarithmic_correctness_test(distributions):
    for idx in range(10):
        res = distributions.Logarithmic("test", 2, 128, 3,
                                        precision=1, group=idx)
        assert res in [100.0, 1e+65, 1e+128]
        assert isinstance(res, float)
