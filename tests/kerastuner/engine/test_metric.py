import pytest
import json
from kerastuner.engine.metric import Metric
import time


@pytest.fixture
def mm():
    mm = Metric('name', 'min')
    mm.update(10)
    mm.update(11)
    return mm


def test_metric_wall_time():
    mm = Metric('acc', 'max')
    mm.update(10)
    time.sleep(1)
    mm.update(11)
    assert mm.wall_time[1] > 1


def test_metric_creation():
    metric = Metric('test', 'min')
    assert metric.name == 'test'
    assert metric.direction == 'min'
    assert int(metric.start_time) == int(time.time())
    assert metric.wall_time == []
    assert metric.history == []


def test_metric_invalid_direction():
    with pytest.raises(ValueError):
        Metric('test', 'invalid')


def test_best_min_value(mm):
    assert mm.get_best_value() == 10


def test_best_max_value():
    mm = Metric('max', 'max')
    mm.update(10)
    mm.update(8)
    assert mm.get_best_value() == 10


def test_last_value(mm):
    assert mm.get_last_value() == 11


def test_get_empty_last_value():
    mm = Metric('min', 'min')
    assert not mm.get_last_value()


def test_update_improve(mm):
    assert mm.update(6)


def test_update_dont_improve(mm):
    assert not mm.update(3713)


def test_single_update():
    mm = Metric('min', 'min')
    assert mm.update(10)


def test_history(mm):
    assert mm.get_history() == [10.0, 11.0]


def test_to_dict(mm):
    start_time = mm.start_time
    conf = mm.to_config()
    assert conf['name'] == 'name'
    assert conf['best_value'] == 10
    assert conf['last_value'] == 11
    assert conf['history'] == [10, 11]
    assert conf['start_time'] == start_time


def test_to_dict_to_json_to_dict(mm):
    conf = mm.to_config()
    conf_json = json.loads(json.dumps(conf))
    assert conf_json == conf


def test_from_config_to_config(mm):
    mm.is_objective = True
    config = mm.to_config()
    mm2 = Metric.from_config(config)
    assert mm2.name == 'name'
    assert mm2.direction == 'min'
    assert mm2.get_history() == [10, 11]
    assert mm2.is_objective
