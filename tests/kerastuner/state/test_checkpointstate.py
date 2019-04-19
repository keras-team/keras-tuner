import pytest
from kerastuner.states.checkpointstate import CheckpointState
from .common import is_serializable


@pytest.fixture
def params():
    return {
        'checkpoint_models': True,
        'checkpoint_monitor': 'loss',
        'checkpoint_mode': 'min'
    }


def test_valid_checkpoint(params):
    cs = CheckpointState(**params)
    assert cs.is_enabled
    assert cs.monitor == 'loss'
    assert cs.mode == 'min'


def test_serializations(params):
    cs = CheckpointState(**params)
    is_serializable(cs)


def test_disable_checkpoint(params):
    params['checkpoint_models'] = False
    cs = CheckpointState(**params)
    assert not cs.is_enabled
    assert not cs.monitor
    assert not cs.mode


def test_bad_mode(params):
    params['checkpoint_mode'] = 'invalid'
    with pytest.raises(ValueError):
        CheckpointState(**params)


def test_bad_monitor(params):
    params['checkpoint_monitor'] = {}
    with pytest.raises(ValueError):
        CheckpointState(**params)
