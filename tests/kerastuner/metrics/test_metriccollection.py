import json
import pytest
import numpy as np
from kerastuner.metrics import MetricCollections, Metric

mc = MetricCollections()


def test_metric_name_add():
    mc.add('acc')
    assert 'acc' in mc.metrics


def test_metric_obj_add():
    mm = Metric('test_add', 'min')
    mc.add(mm)
    assert 'test_add' in mc.metrics


def test_add_invalid_name():
    with pytest.raises(ValueError):
        mc.add('invalid')


def test_duplicate_name():
    mc.add('loss')
    with pytest.raises(ValueError):
        mc.add('loss')


def test_duplicate_obj():
    mm = Metric('dup', 'min')
    mc.add(mm)
    with pytest.raises(ValueError):
        mc.add(mm)


def test_get_metric():
    mm = Metric('to_get', 'min')
    mc.add(mm)
    m2 = mc.get_metric('to_get')
    assert mm == m2


def test_get_metric_invalid():
    assert not mc.get_metric('doesnotexist')


def test_get_metrics():
    col = MetricCollections()
    col.add('loss')
    col.add('acc')
    col.add('val_acc')
    metrics = col.get_metrics()
    assert len(metrics) == 3
    # ensure metrics are properly sorted
    assert metrics[0].name == 'acc'
    assert metrics[2].name == 'val_acc'


def test_update():
    col = MetricCollections()
    col.add('loss')
    assert col.update('loss', 10)


def test_get_dict():
    col = MetricCollections()
    col.add('loss')
    col.add('acc')
    col.add('val_acc')

    col.update('loss', 1)
    col.update('acc', 2)
    col.update('val_loss', 3)

    config = col.get_dict()
    assert config[0]['name'] == 'acc'
    assert config[1]['best_value'] == 1
    assert len(config) == 3


def test_get_dict_to_json():
    col = MetricCollections()
    col.add('loss')
    arr = np.asarray([0.1, 0.2], dtype=np.float32)
    col.update('loss', arr[0])
    config = col.get_dict()
    assert config == json.loads(json.dumps(config))
