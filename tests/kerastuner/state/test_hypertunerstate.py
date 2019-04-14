from __future__ import absolute_import

import pytest
from .common import is_serializable

from kerastuner.states import HypertunerState


def test_is_serializable():
    st = HypertunerState('test')
    is_serializable(st)


def test_invalid_user_info():
    with pytest.raises(ValueError):
        HypertunerState('test', user_info=[])

    with pytest.raises(ValueError):
        HypertunerState('test', user_info='bad')


def test_invalid_epoch_budget():
    with pytest.raises(ValueError):
        HypertunerState('test', epoch_budget=[])


def test_invalid_max_epochs():
    with pytest.raises(ValueError):
        HypertunerState('test', max_epochs=[])