import json
import pytest
import numpy as np
from kerastuner.engine.metric import Metric
from kerastuner.collections import MetricsCollection


@pytest.fixture()
def mc():
    return MetricsCollection()


def test_metric_name_add(mc):
    mc.add('acc')
    metric = mc.get('acc')
    assert isinstance(metric, Metric)
    assert metric.direction == 'max'
    assert metric.name == 'acc'


def test_metric_obj_add(mc):
    mm = Metric('test_add', 'min')
    mc.add(mm)
    metric = mc.get('test_add')
    assert isinstance(metric, Metric)
    assert metric.direction == 'min'
    assert metric.name == 'test_add'
    assert metric == mm


def test_add_invalid_name(mc):
    with pytest.raises(ValueError):
        mc.add('invalid')


def test_duplicate_name(mc):
    mc.add('loss')
    with pytest.raises(ValueError):
        mc.add('loss')


def test_duplicate_obj(mc):
    mm = Metric('dup', 'min')
    mc.add(mm)
    with pytest.raises(ValueError):
        mc.add(mm)


def test_get_metrics(mc):
    mc.add('loss')
    mc.add('acc')
    mc.add('val_acc')
    metrics = mc.to_list()
    assert len(metrics) == 3
    # ensure metrics are properly sorted
    assert metrics[0].name == 'acc'
    assert metrics[1].name == 'loss'
    assert metrics[2].name == 'val_acc'


def test_update_min(mc):
    mc.add('loss')
    # check if update tell us it improved
    assert mc.update('loss', 10)
    # check if update tell us it didn't improve
    assert not mc.update('loss', 12)
    mm = mc.get('loss')
    assert mm.get_best_value() == 10
    assert mm.get_last_value() == 12


def test_update_max(mc):
    mc.add('acc')
    # check if update tell us it improved
    assert mc.update('acc', 10)
    # check if update tell us it didn't improve
    assert mc.update('acc', 12)
    mm = mc.get('acc')
    assert mm.get_best_value() == 12
    assert mm.get_last_value() == 12


def test_to_dict(mc):
    mc.add('loss')
    mc.add('acc')
    mc.add('val_acc')

    mc.update('loss', 1)
    mc.update('acc', 2)
    mc.update('val_loss', 3)

    config = mc.to_dict()
    assert config['acc']['name'] == 'acc'
    assert config['acc']['best_value'] == 2
    assert len(config) == 3


def test_serialization(mc):
    mc.add('loss')
    arr = np.asarray([0.1, 0.2], dtype=np.float32)
    mc.update('loss', arr[0])
    config = mc.to_config()
    assert config == json.loads(json.dumps(config))


def test_set_objective(mc):
    mc.add('loss')
    mc.add('acc')
    mc.add('val_acc')
    mm = mc.get('acc')
    assert not mm.is_objective
    mc.set_objective('acc')
    mm = mc.get('acc')
    assert mm.is_objective


def test_set_invalid_objective(mc):
    with pytest.raises(ValueError):
        mc.set_objective('3713')


def test_double_objective(mc):
    mc.add('loss')
    mc.add('acc')
    mc.set_objective('acc')
    with pytest.raises(ValueError):
        mc.set_objective('loss')
