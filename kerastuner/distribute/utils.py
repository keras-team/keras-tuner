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
"""Distribution utilities."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os


def has_chief_oracle():
    """Checks for distributed tuning with a chief Oracle.

    `CloudOracle` manages its own distribution and so should not set
    "KERASTUNER_ORACLE_IP"

    Returns:
      bool. Whether distributed tuning with a chief Oracle should be run.
    """
    if 'KERASTUNER_ORACLE_IP' in os.environ:
        if 'KERASTUNER_ORACLE_PORT' not in os.environ:
            raise RuntimeError(
                'Environment variable "KERASTUNER_ORACLE_IP" was set, '
                'but "KERASTUNER_ORACLE_PORT" was not. Please specify '
                'a port.')
        if 'KERASTUNER_TUNER_ID' not in os.environ:
            raise RuntimeError(
                'Environment variable "KERASTUNER_ORACLE_IP" was set, '
                'but "KERASTUNER_TUNER_ID" was not. Please specify '
                'an ID for each tuner.')
        return True
    return False


def is_chief_oracle():
    if has_chief_oracle():
        return 'chief' in os.environ['KERASTUNER_TUNER_ID']
    return False
