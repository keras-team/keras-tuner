import pytest
from kerastuner.states.checkpointstate import CheckpointState
from common import is_serializable


def test_valid_checkpoint():
    cs = CheckpointState(True, 'loss', 'min')
    assert cs.is_enabled
    assert cs.monitor == 'loss'
    assert cs.mode == 'min'


def test_serializations():
    cs = CheckpointState(True, 'loss', 'min')
    is_serializable(cs)


def test_disable_checkpoint():
    cs = CheckpointState(False, 'loss', 'min')
    assert not cs.is_enabled
    assert not cs.monitor
    assert not cs.mode


def test_bad_mode():
    with pytest.raises(ValueError):
        cs = CheckpointState(False, 'loss', 'invalid')


def test_bad_monitor():
    with pytest.raises(ValueError):
        cs = CheckpointState(False, None, 'min')
