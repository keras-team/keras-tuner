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
"""Test mock running KerasTuner in a distributed tuning setting."""

import os
import time

import pytest

from keras_tuner.test_utils import mock_distribute


def test_mock_distribute(tmp_path):
    def process_fn():
        assert "KERASTUNER_ORACLE_IP" in os.environ
        # Wait, to test that other threads aren't overriding env vars.
        time.sleep(1)
        assert isinstance(os.environ, mock_distribute.MockEnvVars)
        tuner_id = os.environ["KERASTUNER_TUNER_ID"]
        if "worker" in tuner_id:
            # Give the chief process time to write its value,
            # as we do not join on the chief since it will run
            # a server.
            time.sleep(2)
        fname = os.path.join(str(tmp_path), tuner_id)
        with open(fname, "w") as f:
            f.write(tuner_id)

    mock_distribute.mock_distribute(process_fn, num_workers=3)

    for tuner_id in {"chief", "worker0", "worker1", "worker2"}:
        fname = os.path.join(str(tmp_path), tuner_id)
        with open(fname, "r") as f:
            assert f.read() == tuner_id


def test_exception_raising():
    def worker_error_fn():
        if "worker" in os.environ["KERASTUNER_TUNER_ID"]:
            raise ValueError("Found a worker error")

    with pytest.raises(ValueError, match="Found a worker error"):
        mock_distribute.mock_distribute(worker_error_fn, num_workers=2)

    def chief_error_fn():
        if "chief" in os.environ["KERASTUNER_TUNER_ID"]:
            raise ValueError("Found a chief error")

    with pytest.raises(ValueError, match="Found a chief error"):
        mock_distribute.mock_distribute(chief_error_fn, num_workers=2)
