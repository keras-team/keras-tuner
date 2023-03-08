# Copyright 2019 The KerasTuner Authors
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

import os
from unittest import mock

import pytest

from keras_tuner.distribute import utils
from keras_tuner.test_utils import mock_distribute


def test_no_port_error():
    with mock.patch.object(os, "environ", mock_distribute.MockEnvVars()):
        os.environ["KERASTUNER_ORACLE_IP"] = "127.0.0.1"
        os.environ["KERASTUNER_TUNER_ID"] = "worker0"
        with pytest.raises(RuntimeError, match="KERASTUNER_ORACLE_PORT"):
            utils.has_chief_oracle()


def test_no_id_error():
    with mock.patch.object(os, "environ", mock_distribute.MockEnvVars()):
        os.environ["KERASTUNER_ORACLE_IP"] = "127.0.0.1"
        os.environ["KERASTUNER_ORACLE_PORT"] = "80"
        with pytest.raises(RuntimeError, match="KERASTUNER_TUNER_ID"):
            utils.has_chief_oracle()
