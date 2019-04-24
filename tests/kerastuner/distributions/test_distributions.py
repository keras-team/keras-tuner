import pytest
from kerastuner.distributions.distributions import Distributions


def test_record_retrieve_param():
    dist = Distributions('test', {})
    name = 'param1'
    value = 3713
    group = 'group'
    key = dist._get_key(name, group)

    # record
    dist._record_hyperparameter(name, value, group)

    # retrieve
    hparams = dist.get_hyperparameters()
    assert key in hparams
    assert hparams[key]['name'] == name
    assert hparams[key]['value'] == value
    assert hparams[key]['group'] == group


def test_record_retrieve_config():
    hparam_config = {
        "this": 'that'
    }

    dist = Distributions('test', hparam_config)
    hparam_config2 = dist.get_hyperparameters_config()
    assert hparam_config == hparam_config2
