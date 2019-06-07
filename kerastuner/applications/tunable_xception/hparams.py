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

from kerastuner.distributions import Choice, Linear, Range


def default_fixed_hparams(input_shape, num_classes):
    hp = {
        "kernel_size": (3, 3),
        "initial_strides": (2, 2),
        "activation": "relu",
        "learning_rate": .001,
        "conv2d_num_filters": 64,
        "sep_num_filters": 256,
        "num_residual_blocks": 4,
        "dense_use_bn": True,
        "dropout_rate": 0.0,
        "dense_merge_type": "avg",
        "num_dense_layers": 1
    }
    return hp


def default_hparams(input_shape, num_classes):

    hp = {}

    # [general]
    kernel_size = Range("kernel_size", 3, 5, 2, group="general")

    hp["kernel_size"] = (kernel_size, kernel_size)
    hp["initial_strides"] = (2, 2)
    hp["activation"] = Choice("activation", ["relu", "selu"], group="general")
    hp["learning_rate"] = Choice("learning_rate", [.001, .0001, .00001],
                                 group="general")

    # [entry flow]

    # -conv2d
    hp["conv2d_num_filters"] = Choice(
        "num_filters", [32, 64, 128], group="conv2d")

    # seprarable block > not an exact match to the paper
    hp["sep_num_filters"] = Range(
        "num_filters", 128, 768, 128, group="entry_flow")

    # [Middle Flow]
    hp["num_residual_blocks"] = Range("num_residual_blocks", 2, 8,
                                      group="middle_flow")

    # [Exit Flow]
    hp["dense_merge_type"] = Choice("merge_type", ["avg", "flatten", "max"],
                                    group="exit_flow")

    hp["num_dense_layers"] = Range("dense_layers", 1, 3, group="exit_flow")

    hp["dropout_rate"] = Linear("dropout", start=0.0, stop=0.5, num_buckets=6,
                                precision=1, group="exit_flow")
    hp["dense_use_bn"] = Choice("batch_normalization", [True, False],
                                "exit_flow")
    return hp
