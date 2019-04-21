import pytest
from kerastuner.collections.collections import Collection


@pytest.fixture
def obj():
    return {
        'idx': '3713',
    }


def test_len():
    c = Collection()
    c.add('a', 'a')
    c.add('b', 'b')
    c.add('c', 'c')
    assert len(c) == 3


def test_add_get(obj):
    ic = Collection()
    idx = obj['idx']
    ic.add(idx, obj)
    instance2 = ic.get(idx)
    assert instance2 == obj


def test_add_get_last(obj):
    ic = Collection()
    idx = obj['idx']
    ic.add(idx, obj)
    instance2 = ic.get_last()
    assert instance2 == obj
