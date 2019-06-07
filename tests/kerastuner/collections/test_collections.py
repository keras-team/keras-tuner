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

import pytest
from kerastuner.collections.collections import Collection


@pytest.fixture
def obj():
    return {
        'idx': '3713',
    }


def test_len():
    c = Collection()
    c.add('a', 'a')
    c.add('b', 'b')
    c.add('c', 'c')
    assert len(c) == 3


def test_empty_get_last():
    ic = Collection()
    assert not ic.get_last()


def test_add_get(obj):
    ic = Collection()
    idx = obj['idx']
    ic.add(idx, obj)
    instance2 = ic.get(idx)
    assert instance2 == obj


def test_add_get_last(obj):
    ic = Collection()
    idx = obj['idx']
    ic.add(idx, obj)
    instance2 = ic.get_last()
    assert instance2 == obj


def test_get_metric_invalid():
    ic = Collection()
    assert not ic.get('doesnotexist')


def test_exist():
    ic = Collection()
    ic.add('a', 'a')
    assert ic.exist('a')
    assert not ic.exist('b')


def test_update(obj):
    ic = Collection()
    idx = obj['idx']
    ic.add(idx, obj)
    ic.update(idx, {'idx': 42})
    obj2 = ic.get(idx)
    assert obj2['idx'] == 42


def test_to_list():
    c = Collection()
    c.add('a', 'a')
    c.add('b', 'b')
    c.add('c', 'c')
    lst = c.to_list()
    assert lst[0] == 'a'
    assert lst[1] == 'b'
    assert lst[2] == 'c'


def test_reverse_list():
    c = Collection()
    c.add('a', 'a')
    c.add('b', 'b')
    c.add('c', 'c')
    lst = c.to_list(reverse=True)
    assert lst[0] == 'c'
    assert lst[1] == 'b'
    assert lst[2] == 'a'


def test_to_dict():
    c = Collection()
    c.add('a', 'a')
    c.add('b', 'b')
    c.add('c', 'c')
    d = c.to_dict()
    assert d['a'] == 'a'
    assert d['b'] == 'b'
    assert d['c'] == 'c'
