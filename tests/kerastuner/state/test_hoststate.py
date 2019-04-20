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


def test_summary(kwargs, capsys):
    cs = HostState(**kwargs)
    cs.summary()
    captured = capsys.readouterr()
    to_test = [
        'results: %s' % kwargs.get('result_dir'),
        'tmp: %s' % kwargs.get('tmp_dir'),
        'export: %s' % kwargs.get('export_dir'),
    ]
    for s in to_test:
        assert s in captured.out


def test_extended_summary_working(kwargs, capsys):
    cs = HostState(**kwargs)
    cs.summary()
    summary_out = capsys.readouterr()
    cs.summary(extended=True)
    extended_summary_out = capsys.readouterr()
    assert summary_out.out.count(":") < extended_summary_out.out.count(":")
