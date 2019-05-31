# Copyright 2019 The Keras Tuner Authors
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     https://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import

import pytest
from .common import is_serializable

from kerastuner.states import TunerState


@pytest.fixture
def kwargs(tmpdir):
    kwargs = {
        "results_dir": str(tmpdir + '/results/'),
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


def test_summary(kwargs, capsys):
    state = TunerState('test', None, **kwargs)
    state.summary()
    captured = capsys.readouterr()
    to_test = [
        'results: %s' % kwargs.get('results_dir'),
        'tmp: %s' % kwargs.get('tmp_dir'),
        'export: %s' % kwargs.get('export_dir'),
    ]
    for s in to_test:
        assert s in captured.out


def test_extended_summary_working(kwargs, capsys):
    state = TunerState('test', None, **kwargs)
    state.summary()
    summary_out = capsys.readouterr()
    state.summary(extended=True)
    extended_summary_out = capsys.readouterr()
    assert summary_out.out.count(":") < extended_summary_out.out.count(":")
