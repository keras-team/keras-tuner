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

"""KerasTuner protos."""

from contextlib import contextmanager

import sys


@contextmanager
def protobuf_check():
    try:
        yield
    except ImportError:
        from google import protobuf

        raise ImportError(
            "keras_tuner parallel tuning requires protobuf>=4, "
            f"but got protobuf=={protobuf.__version__}."
        )  # pragma: no cover


def get_proto():
    with protobuf_check():
        from keras_tuner.protos import keras_tuner_pb2
    return keras_tuner_pb2


def get_service():
    with protobuf_check():
        from keras_tuner.protos import service_pb2
    return service_pb2


def get_service_grpc():
    with protobuf_check():
        from keras_tuner.protos import service_pb2_grpc
    return service_pb2_grpc
