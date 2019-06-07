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

from collections import defaultdict
import pytest

from kerastuner.distributions import SequentialDistributions
from .common import record_hyperparameters_test, json_serialize_test
from .common import fixed_correctness_test, bool_correctness_test
from .common import choice_correctness_test, range_type_correctness_test
from .common import linear_correctness_test, logarithmic_correctness_test


HPARAMS = {
    "default:choice": {
        "name": "choice",
        "group": "default",
        "type": "Choice",
        "space_size": 3,
        "start": 1,
        "stop": 3,
        "values": [1, 2, 3]
    },
    "group2:choice": {
        "name": "choice",
        "group": "default",
        "type": "Choice",
        "space_size": 3,
        "start": 1,
        "stop": 3,
        "values": [1, 2, 3]
    }
}


def test_record_hyperparameters():

    seq = SequentialDistributions(HPARAMS)
    hp_curr = seq.get_hyperparameters_config()

    assert len(hp_curr) == 2
    k = seq._get_key('choice', 'default')
    assert hp_curr[k]['name'] == 'choice'
    assert hp_curr[k]['group'] == 'default'
    assert hp_curr[k]['values'] == [1, 2, 3]


# check proper use of name and group
def test_naming():
    seq = SequentialDistributions(HPARAMS)
    assert seq.Choice('choice', [1, 2, 3]) == 1
    assert seq.Choice('choice', [1, 2, 3], group='group2') == 1
    assert seq.Choice('choice', [1, 2, 3]) == 2


def test_sequencing():
    seq = SequentialDistributions(HPARAMS)

    # Cycle 1
    assert seq.Choice('choice', [1, 2, 3]) == 1
    assert seq.Choice('choice', [1, 2, 3], group='group2') == 1

    assert seq.Choice('choice', [1, 2, 3]) == 2
    assert seq.Choice('choice', [1, 2, 3], group='group2') == 1
    assert seq.Choice('choice', [1, 2, 3]) == 3
    assert seq.Choice('choice', [1, 2, 3], group='group2') == 1

    # Cycle 2
    assert seq.Choice('choice', [1, 2, 3]) == 1
    assert seq.Choice('choice', [1, 2, 3], group='group2') == 2

    assert seq.Choice('choice', [1, 2, 3]) == 2
    assert seq.Choice('choice', [1, 2, 3], group='group2') == 2

    assert seq.Choice('choice', [1, 2, 3]) == 3
    assert seq.Choice('choice', [1, 2, 3], group='group2') == 2

    # Cycle 3
    assert seq.Choice('choice', [1, 2, 3]) == 1
    assert seq.Choice('choice', [1, 2, 3], group='group2') == 3

    assert seq.Choice('choice', [1, 2, 3]) == 2
    assert seq.Choice('choice', [1, 2, 3], group='group2') == 3

    assert seq.Choice('choice', [1, 2, 3]) == 3
    assert seq.Choice('choice', [1, 2, 3], group='group2') == 3

    # Cycle 1 again
    assert seq.Choice('choice', [1, 2, 3]) == 1
    assert seq.Choice('choice', [1, 2, 3], group='group2') == 1
