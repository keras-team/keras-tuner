import pytest
from kerastuner.state.checkpointstate import CheckpointState


def test_valid_checkpoint():
    cs = CheckpointState(True, 'loss', 'min')
    assert cs.is_enabled
    assert cs.monitor == 'loss'
    assert cs.mode == 'min'


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
