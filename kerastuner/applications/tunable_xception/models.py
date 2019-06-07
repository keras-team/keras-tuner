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

import copy
import functools

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

from kerastuner.applications.tunable_xception.blocks import sep_conv, conv, dense, residual
from kerastuner.applications.tunable_xception.hparams import default_fixed_hparams, default_hparams


def TunableXception(input_shape, num_classes, **hparams):
    """ Returns a wrapper around a tunable_xception function which provides the
    specified shape, number of output classes, and hyperparameters.

    Args:
        input_shape (tuple):  Shape of the input image.
        num_classes (int): Number of output classes.
        hparams (**dictionary): Hyperparameters to the TunableXception model.

    Returns:
        A wrapped tunable_xception function.
    """

    return functools.partial(_xception, input_shape, num_classes, **hparams)


def tunable_xception_single_fn(input_shape, num_classes, **hparams):
    """ Returns a wrapper around a tunable_xception_single function which provides the
    specified shape, number of output classes, and hyperparameters.

    Args:
        input_shape (tuple):  Shape of the input image.
        num_classes (int): Number of output classes.
        hparams (**dictionary): Hyperparameters to the TunableXception model.

    Returns:
        A wrapped tunable_xception_single function.
    """

    return functools.partial(_xception_single, input_shape, num_classes, **hparams)


def tunable_xception_single_model(input_shape, num_classes, **hparams):
    model_fn = tunable_xception_single_fn(input_shape, num_classes, **hparams)
    i = model_fn()
    return i


def _xception(input_shape, num_classes, **hparams):
    """
    Implementation of a hypertunable adaptation of Xception.
    """
    hp = {}
    hp.update(default_hparams(input_shape, num_classes))
    if hparams:
        hp.update(hparams)

    ### Parameters ###

    # [general]
    kernel_size = hp["kernel_size"]

    initial_strides = hp["initial_strides"]
    activation = hp["activation"]

    # [entry flow]

    # -conv2d
    conv2d_num_filters = hp["conv2d_num_filters"]

    # seprarable block > not an exact match to the paper
    sep_num_filters = hp["sep_num_filters"]

    # [Middle Flow]
    num_residual_blocks = hp["num_residual_blocks"]

    # [Exit Flow]
    dense_merge_type = hp["dense_merge_type"]
    num_dense_layers = hp["num_dense_layers"]
    dropout_rate = hp["dropout_rate"]
    dense_use_bn = hp["dense_use_bn"]

    ### Model ###
    # input
    inputs = keras.Input(shape=input_shape)
    x = inputs

    # Initial conv2d
    dims = conv2d_num_filters
    x = conv(x, dims, kernel_size=kernel_size, activation=activation,
             strides=initial_strides)

    # separable convs
    dims = sep_num_filters
    for _ in range(num_residual_blocks):
        x = residual(x, dims, activation=activation, max_pooling=False)

    # Exit
    dims *= 2
    x = residual(x, dims, activation=activation, max_pooling=True)

    if dense_merge_type == 'flatten':
        x = layers.Flatten()(x)
    elif dense_merge_type == "avg":
        x = layers.GlobalAveragePooling2D()(x)
    else:
        x = layers.GlobalMaxPooling2D()(x)

    # Dense
    for _ in range(num_dense_layers):
        x = dense(x, num_classes, activation=activation, batchnorm=dense_use_bn,
                  dropout_rate=dropout_rate)
    output = layers.Dense(num_classes, activation='softmax')(x)
    model = keras.Model(inputs, output)

    lr = hp["learning_rate"]
    optimizer = Adam(lr=lr)

    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy'])

    return model


def _xception_single(
        input_shape,
        num_classes,
        mode="full",
        **hparams):
    """ Model fn which uses a single value for the hyper parameters """

    hp = default_fixed_hparams(input_shape, num_classes)
    hp.update(hparams)
    return _xception(input_shape=input_shape, num_classes=num_classes, **hp)
