import pytest
from kerastuner.engine.instancecollection import InstanceCollection


@pytest.fixture
def instance():
    return {
        'idx': '3713',
    }


def test_add_get(instance):
    ic = InstanceCollection()
    idx = instance['idx']
    ic.add(idx, instance)
    instance2 = ic.get(idx)
    assert instance2 == instance


def test_add_get_last(instance):
    ic = InstanceCollection()
    idx = instance['idx']
    ic.add(idx, instance)
    instance2 = ic.get_last()
    assert instance2 == instance
