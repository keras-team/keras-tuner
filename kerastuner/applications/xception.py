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

import tensorflow.keras as keras
from tensorflow.keras import layers

from kerastuner.engine import hypermodel


class HyperXception(hypermodel.HyperModel):
    """An Xception HyperModel."""

    def __init__(self, input_shape, num_classes, include_top=True):
        super(HyperXception, self).__init__()
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.include_top = include_top

    def build(self, hp):
        activation = hp.Choice('activation', ['relu', 'selu'])

        # Model definition.
        inputs = keras.Input(shape=self.input_shape)
        x = inputs

        # Initial conv2d.
        conv2d_num_filters = hp.Choice(
            'conv2d_num_filters', [32, 64, 128], default=64)
        kernel_size = hp.Choice('kernel_size', [3, 5])
        initial_strides = hp.Choice('initial_strides', [2])
        x = conv(x,
                 conv2d_num_filters,
                 kernel_size=kernel_size,
                 activation=activation,
                 strides=initial_strides)

        # Separable convs.
        sep_num_filters = hp.Range(
            'sep_num_filters', 128, 768, step=128, default=256)
        num_residual_blocks = hp.Range('num_residual_blocks', 2, 8, default=4)
        for _ in range(num_residual_blocks):
            x = residual(x,
                         sep_num_filters,
                         activation=activation,
                         max_pooling=False)
        # Exit flow.
        x = residual(x,
                     2*sep_num_filters,
                     activation=activation,
                     max_pooling=True)

        pooling = hp.Choice('pooling', ['avg', 'flatten', 'max'])
        if pooling == 'flatten':
            x = layers.Flatten()(x)
        elif pooling == 'avg':
            x = layers.GlobalAveragePooling2D()(x)
        else:
            x = layers.GlobalMaxPooling2D()(x)

        if self.include_top:
            # Dense
            num_dense_layers = hp.Range('num_dense_layers', 1, 3)
            dropout_rate = hp.Linear(
                'dropout_rate', 0.0, 0.6, resolution=0.1, default=0.5)
            dense_use_bn = hp.Choice('dense_use_bn', [True, False])
            for _ in range(num_dense_layers):
                x = dense(x,
                          self.num_classes,
                          activation=activation,
                          batchnorm=dense_use_bn,
                          dropout_rate=dropout_rate)
            output = layers.Dense(self.num_classes, activation='softmax')(x)
            model = keras.Model(inputs, output, name='Xception')

            model.compile(
                optimizer=keras.optimizers.Adam(
                    hp.Choice('learning_rate', [1e-3, 1e-4, 1e-5])),
                loss='categorical_crossentropy',
                metrics=['accuracy'])
            return model
        else:
            return keras.Model(inputs, x, name='Xception')


def sep_conv(x, num_filters, kernel_size=(3, 3), activation='relu'):
    if activation == 'selu':
        x = layers.SeparableConv2D(num_filters, kernel_size,
                                   activation='selu',
                                   padding='same',
                                   kernel_initializer='lecun_normal')(x)
    elif activation == 'relu':
        x = layers.SeparableConv2D(num_filters, kernel_size,
                                   padding='same',
                                   use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
    else:
        ValueError('Unkown activation function: %s' % (activation,))
    return x


def residual(x, num_filters,
             kernel_size=(3, 3),
             activation='relu',
             pool_strides=(2, 2),
             max_pooling=True):
    "Residual block."
    if max_pooling:
        res = layers.Conv2D(num_filters, kernel_size=(
            1, 1), strides=pool_strides, padding='same')(x)
    elif num_filters != keras.backend.int_shape(x)[-1]:
        res = layers.Conv2D(num_filters, kernel_size=(1, 1), padding='same')(x)
    else:
        res = x

    x = sep_conv(x, num_filters, kernel_size, activation)
    x = sep_conv(x, num_filters, kernel_size, activation)
    if max_pooling:
        x = layers.MaxPooling2D(
            kernel_size, strides=pool_strides, padding='same')(x)

    x = layers.add([x, res])
    return x


def conv(x, num_filters,
         kernel_size=(3, 3), activation='relu', strides=(2, 2)):
    "2d convolution block."
    if activation == 'selu':
        x = layers.Conv2D(num_filters, kernel_size,
                          strides=strides, activation='selu',
                          padding='same', kernel_initializer='lecun_normal',
                          bias_initializer='zeros')(x)
    elif activation == 'relu':
        x = layers.Conv2D(num_filters, kernel_size,
                          strides=strides, padding='same', use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
    else:
        msg = 'Unkown activation function: %s' % activation
        ValueError(msg)
    return x


def dense(x, dims, activation='relu', batchnorm=True, dropout_rate=0):
    if activation == 'selu':
        x = layers.Dense(dims,  activation='selu',
                         kernel_initializer='lecun_normal',
                         bias_initializer='zeros')(x)
        if dropout_rate:
            x = layers.AlphaDropout(dropout_rate)(x)
    elif activation == 'relu':
        x = layers.Dense(dims, activation='relu')(x)
        if batchnorm:
            x = layers.BatchNormalization()(x)
        if dropout_rate:
            x = layers.Dropout(dropout_rate)(x)
    else:
        msg = 'Unkown activation function: %s' % activation
        ValueError(msg)
    return x
