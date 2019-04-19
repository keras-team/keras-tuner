from __future__ import absolute_import

import pytest
from .common import is_serializable

from kerastuner.states import TunerState


def test_is_serializable():
    st = TunerState('test', None)
    is_serializable(st)


def test_invalid_user_info():
    with pytest.raises(ValueError):
        TunerState('test', None, user_info=[])

    with pytest.raises(ValueError):
        TunerState('test', None, user_info='bad')


def test_invalid_epoch_budget():
    with pytest.raises(ValueError):
        TunerState('test', None, epoch_budget=[])


def test_invalid_max_epochs():
    with pytest.raises(ValueError):
        TunerState('test', None, max_epochs=[])
