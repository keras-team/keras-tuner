import pytest
import os
from .common import is_serializable

from kerastuner.states.hoststate import HostState


@pytest.fixture
def kwargs(tmpdir):
    kwargs = {
        "result_dir": str(tmpdir + '/results/'),
        "tmp_dir": str(tmpdir + '/tmp/'),
        "export_dir": str(tmpdir + '/export/')
    }
    return kwargs


def test_is_serializable(kwargs):
    st = HostState(**kwargs)
    is_serializable(st)


def test_dir_creation(kwargs):
    HostState(**kwargs)
    assert os.path.exists(kwargs.get('result_dir'))
    assert os.path.exists(kwargs.get('tmp_dir'))
    assert os.path.exists(kwargs.get('export_dir'))
