from __future__ import absolute_import

import pytest
from .common import is_serializable

from kerastuner.states import TunerState


@pytest.fixture
def kwargs(tmpdir):
    kwargs = {
        "result_dir": str(tmpdir + '/results/'),
        "tmp_dir": str(tmpdir + '/tmp/'),
        "export_dir": str(tmpdir + '/export/')
    }
    return kwargs


def test_is_serializable(kwargs):
    st = TunerState('test', None, **kwargs)
    is_serializable(st)


def test_invalid_user_info(kwargs):
    with pytest.raises(ValueError):
        TunerState('test', None, user_info=[], **kwargs)

    with pytest.raises(ValueError):
        TunerState('test', None, user_info='bad', **kwargs)


# FIXME: test negative budget, min >> max -- max >> total
def test_invalid_epoch_budget(kwargs):

    with pytest.raises(ValueError):
        TunerState('test', None, epoch_budget=[], **kwargs)


def test_invalid_max_epochs(kwargs):

    with pytest.raises(ValueError):
        TunerState('test', None, max_epochs=[], **kwargs)
