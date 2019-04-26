import json
import pytest
import numpy as np
from kerastuner.engine.metric import Metric
from kerastuner.collections import MetricsCollection


@pytest.fixture()
def mc():
    return MetricsCollection()


def test_metric_name_add(mc):
    mc.add('accuracy')
    metric = mc.get('accuracy')
    assert isinstance(metric, Metric)
    assert metric.direction == 'max'
    assert metric.name == 'accuracy'


def test_metric_obj_add(mc):
    mm = Metric('test', 'min')
    mc.add(mm)
    metric = mc.get('test')
    assert isinstance(metric, Metric)
    assert metric.direction == 'min'
    assert metric.name == 'test'
    assert mm == metric


def test_add_invalid_name(mc):
    with pytest.raises(ValueError):
        mc.add('invalid')


def test_duplicate_name(mc):
    mc.add('loss')
    with pytest.raises(ValueError):
        mc.add('loss')


def test_duplicate_obj(mc):
    mm = Metric('acc', 'min')
    mc.add(mm)
    with pytest.raises(ValueError):
        mc.add(mm)


def test_get_metrics(mc):
    mc.add('loss')
    mc.add('accuracy')
    mc.add('val_accuracy')
    metrics = mc.to_list()
    assert len(metrics) == 3
    # ensure metrics are properly sorted
    assert metrics[0].name == 'accuracy'
    assert metrics[1].name == 'loss'
    assert metrics[2].name == 'val_accuracy'


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
    mc.add('accuracy')
    # check if update tell us it improved
    assert mc.update('accuracy', 10)
    # check if update tell us it didn't improve
    assert mc.update('accuracy', 12)
    mm = mc.get('accuracy')
    assert mm.get_best_value() == 12
    assert mm.get_last_value() == 12


def test_to_dict(mc):
    mc.add('loss')
    mc.add('accuracy')
    mc.add('val_accuracy')

    mc.update('loss', 1)
    mc.update('accuracy', 2)
    mc.update('val_loss', 3)

    config = mc.to_dict()
    assert config['accuracy'].name == 'accuracy'
    assert config['accuracy'].get_best_value() == 2
    assert len(config) == 3


def test_serialization(mc):
    mc.add('loss')
    arr = np.asarray([0.1, 0.2], dtype=np.float32)
    mc.update('loss', arr[0])
    config = mc.to_config()
    assert config == json.loads(json.dumps(config))


def test_alias(mc):
    mc.add('acc')
    mm = mc.get('acc')
    assert mm.name == 'accuracy'


def test_alias_update(mc):
    mc.add('acc')
    mc.update('acc', 14)
    mc.update('acc', 12)
    mm = mc.get('acc')
    assert mm.history == [14, 12]
    assert mm.get_best_value() == 14
    assert mm.get_last_value() == 12


def test_set_objective(mc):
    mc.add('loss')
    mc.add('accuracy')
    mc.add('val_accuracy')
    mm = mc.get('accuracy')
    assert not mm.is_objective
    mc.set_objective('accuracy')
    mm = mc.get('accuracy')
    assert mm.is_objective


def test_set_invalid_objective(mc):
    with pytest.raises(ValueError):
        mc.set_objective('3713')


def test_set_shortand_objective(mc):
    mc.add('accuracy')
    mc.set_objective('acc')
    assert mc._objective_name == 'accuracy'


def test_set_shortand_val_objective(mc):
    mc.add('val_accuracy')
    mc.set_objective('val_acc')
    assert mc._objective_name == 'val_accuracy'


def test_double_objective(mc):
    mc.add('loss')
    mc.add('accuracy')
    mc.set_objective('accuracy')
    with pytest.raises(ValueError):
        mc.set_objective('loss')


def test_from_config_to_config(mc):
    config = mc.to_config()
    mc2 = MetricsCollection.from_config(config)
    mcl = mc.to_list()
    mc2l = mc2.to_list()

    assert mc2._objective_name == mc._objective_name
    for idx in range(len(mcl)):
        assert mcl[idx].name == mc2l[idx].name
