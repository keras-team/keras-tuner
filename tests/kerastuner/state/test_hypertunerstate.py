from __future__ import absolute_import

import pytest
from .common import is_serializable, exportable_attributes_exists

from kerastuner.states import HypertunerState


def test_exportable_attributes():
    st = HypertunerState('test', {})
    exportable_attributes_exists(st)


def test_is_serializable():
    st = HypertunerState('test', {})
    is_serializable(st)


def test_invalid_user_info():
    with pytest.raises(ValueError):
        st = HypertunerState('test', [])
