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

import pytest

from keras_tuner import utils


def test_check_tf_version_error():
    utils.tf.__version__ = "1.15.0"

    with pytest.warns(ImportWarning) as record:
        utils.check_tf_version()
    assert len(record) == 1
    assert (
        "Tensorflow package version needs to be at least 2.0.0"
        in record[0].message.args[0]
    )


def test_to_list_with_tuple_return_list():
    result = utils.to_list((1, 2, 3))
    assert isinstance(result, list)
    assert result == [1, 2, 3]


def test_try_clear_without_ipython():
    is_notebook = utils.IS_NOTEBOOK
    utils.IS_NOTEBOOK = False
    utils.try_clear()
    utils.IS_NOTEBOOK = is_notebook


def test_create_directory_and_remove_existing(tmp_path):
    utils.create_directory(tmp_path, remove_existing=True)
