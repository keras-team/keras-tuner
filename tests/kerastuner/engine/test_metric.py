import pytest
import json
from kerastuner.engine.metric import Metric


def test_metric_creation():
    metric = Metric('test', 'min')
    assert metric.name == 'test'
    assert metric.direction == 'min'


def test_metric_invalid_direction():
    with pytest.raises(ValueError):
        Metric('test', 'invalid')


def test_best_value():
    mm = Metric('min', 'min')
    mm.update(10)
    mm.update(8)
    assert mm.get_best_value() == 8

    mm = Metric('max', 'max')
    mm.update(10)
    mm.update(8)
    assert mm.get_best_value() == 10


def test_last_value():
    mm = Metric('min', 'min')
    mm.update(10)
    mm.update(8)
    assert mm.get_last_value() == 8


def test_get_empty_last_value():
    mm = Metric('min', 'min')
    assert not mm.get_last_value()


def test_update():
    mm = Metric('min', 'min')
    mm.update(10)
    assert mm.update(8)


def test_single_update():
    mm = Metric('min', 'min')
    assert mm.update(10)


def test_history():
    mm = Metric('min', 'min')
    mm.update(10)
    mm.update(8)
    assert mm.get_history() == [10, 8]


def test_to_dict():
    mm = Metric('conf_test', 'min')
    mm.update(8)
    mm.update(10)
    conf = mm.to_config()
    assert conf['name'] == 'conf_test'
    assert conf['best_value'] == 8
    assert conf['last_value'] == 10
    assert conf['history'] == [8, 10]


def test_to_dict_to_json_to_dict():
    mm = Metric('min', 'min')
    mm.update(10)
    mm.update(8)
    conf = mm.to_config()
    conf_json = json.loads(json.dumps(conf))
    assert conf_json == conf
