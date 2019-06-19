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

"""Hypertunable version of Resnet."""

import tensorflow.keras as keras
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras import backend

from kerastuner.engine import hypermodel


class HyperResnet(hypermodel.HyperModel):
    """A ResNet HyperModel."""

    def __init__(self, input_shape, num_classes, include_top=True):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.include_top = include_top

    def build(self, hp):
        version = hp.Choice('version', ['v1', 'v2', 'next'], default='v2')

        # Version-conditional hyperparameters.
        with hp.name_scope(version):
            conv3_depth = hp.Choice(
                'conv3_depth',
                [4] if version == 'next' else [4, 8],
                default=4)
            conv4_depth = hp.Choice(
                'conv4_depth',
                [6, 23] if version == 'next' else [6, 23, 36],
                default=6)

        # Version-conditional fixed parameters
        preact = True if version == 'v2' else False
        use_bias = False if version == 'next' else True

        # Model definition.
        bn_axis = 3 if backend.image_data_format() == 'channels_last' else 1
        inputs = layers.Input(shape=self.input_shape)
        x = inputs

        # Initial conv2d block.
        x = layers.ZeroPadding2D(padding=((3, 3), (3, 3)), name='conv1_pad')(x)
        x = layers.Conv2D(
            64, 7, strides=2, use_bias=use_bias, name='conv1_conv')(x)
        if preact is False:
            x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                                        name='conv1_bn')(x)
            x = layers.Activation('relu', name='conv1_relu')(x)
        x = layers.ZeroPadding2D(padding=((1, 1), (1, 1)), name='pool1_pad')(x)
        x = layers.MaxPooling2D(3, strides=2, name='pool1_pool')(x)

        # Middle hypertunable stack.
        if version == 'v1':
            x = stack1(x, 64, 3, stride1=1, name='conv2')
            x = stack1(x, 128, conv3_depth, name='conv3')
            x = stack1(x, 256, conv4_depth, name='conv4')
            x = stack1(x, 512, 3, name='conv5')
        elif version == 'v2':
            x = stack2(x, 64, 3, name='conv2')
            x = stack2(x, 128, conv3_depth, name='conv3')
            x = stack2(x, 256, conv4_depth, name='conv4')
            x = stack2(x, 512, 3, stride1=1, name='conv5')
        elif version == 'next':
            x = stack3(x, 64, 3, name='conv2')
            x = stack3(x, 256, conv3_depth, name='conv3')
            x = stack3(x, 512, conv4_depth, name='conv4')
            x = stack3(x, 1024, 3, stride1=1, name='conv5')

        # Top of the model.
        if preact is True:
            x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                                        name='post_bn')(x)
            x = layers.Activation('relu', name='post_relu')(x)

        if self.include_top:
            x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
            x = layers.Dense(self.num_classes, activation='softmax', name='probs')(x)
        else:
            pooling = hp.Choice('pooling', ['avg', 'max'], default='avg')
            if pooling == 'avg':
                x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
            elif pooling == 'max':
                x = layers.GlobalMaxPooling2D(name='max_pool')(x)

        model = keras.Model(inputs, x, name='Resnet')

        optimizer_name = hp.Choice('optimizer', ['adam', 'rmsprop', 'sgd'], default='adam')
        optimizer = keras.optimizers.get(optimizer_name)
        optimizer.learning_rate = hp.Choice('learning_rate', [0.1, 0.01, 0.001], default=0.01)
        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy'])
        return model


def block1(x, filters, kernel_size=3, stride=1,
           conv_shortcut=True, name=None):
    """A residual block.
    # Arguments
        x: input tensor.
        filters: integer, filters of the bottleneck layer.
        kernel_size: default 3, kernel size of the bottleneck layer.
        stride: default 1, stride of the first layer.
        conv_shortcut: default True, use convolution shortcut if True,
            otherwise identity shortcut.
        name: string, block label.
    # Returns
        Output tensor for the residual block.
    """
    bn_axis = 3 if backend.image_data_format() == 'channels_last' else 1

    if conv_shortcut is True:
        shortcut = layers.Conv2D(4 * filters, 1, strides=stride,
                                 name=name + '_0_conv')(x)
        shortcut = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                                             name=name + '_0_bn')(shortcut)
    else:
        shortcut = x

    x = layers.Conv2D(filters, 1, strides=stride, name=name + '_1_conv')(x)
    x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                                  name=name + '_1_bn')(x)
    x = layers.Activation('relu', name=name + '_1_relu')(x)

    x = layers.Conv2D(filters, kernel_size, padding='same',
                      name=name + '_2_conv')(x)
    x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                                  name=name + '_2_bn')(x)
    x = layers.Activation('relu', name=name + '_2_relu')(x)

    x = layers.Conv2D(4 * filters, 1, name=name + '_3_conv')(x)
    x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                                  name=name + '_3_bn')(x)

    x = layers.Add(name=name + '_add')([shortcut, x])
    x = layers.Activation('relu', name=name + '_out')(x)
    return x


def stack1(x, filters, blocks, stride1=2, name=None):
    """A set of stacked residual blocks.
    # Arguments
        x: input tensor.
        filters: integer, filters of the bottleneck layer in a block.
        blocks: integer, blocks in the stacked blocks.
        stride1: default 2, stride of the first layer in the first block.
        name: string, stack label.
    # Returns
        Output tensor for the stacked blocks.
    """
    x = block1(x, filters, stride=stride1, name=name + '_block1')
    for i in range(2, blocks + 1):
        x = block1(
            x,
            filters,
            conv_shortcut=False,
            name=name +
            '_block' +
            str(i))
    return x


def block2(x, filters, kernel_size=3, stride=1,
           conv_shortcut=False, name=None):
    """A residual block.
    # Arguments
        x: input tensor.
        filters: integer, filters of the bottleneck layer.
        kernel_size: default 3, kernel size of the bottleneck layer.
        stride: default 1, stride of the first layer.
        conv_shortcut: default False, use convolution shortcut if True,
            otherwise identity shortcut.
        name: string, block label.
    # Returns
        Output tensor for the residual block.
    """
    bn_axis = 3 if backend.image_data_format() == 'channels_last' else 1

    preact = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                                       name=name + '_preact_bn')(x)
    preact = layers.Activation('relu', name=name + '_preact_relu')(preact)

    if conv_shortcut is True:
        shortcut = layers.Conv2D(4 * filters, 1, strides=stride,
                                 name=name + '_0_conv')(preact)
    else:
        shortcut = layers.MaxPooling2D(
            1, strides=stride)(x) if stride > 1 else x

    x = layers.Conv2D(filters, 1, strides=1, use_bias=False,
                      name=name + '_1_conv')(preact)
    x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                                  name=name + '_1_bn')(x)
    x = layers.Activation('relu', name=name + '_1_relu')(x)

    x = layers.ZeroPadding2D(padding=((1, 1), (1, 1)), name=name + '_2_pad')(x)
    x = layers.Conv2D(filters, kernel_size, strides=stride,
                      use_bias=False, name=name + '_2_conv')(x)
    x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                                  name=name + '_2_bn')(x)
    x = layers.Activation('relu', name=name + '_2_relu')(x)

    x = layers.Conv2D(4 * filters, 1, name=name + '_3_conv')(x)
    x = layers.Add(name=name + '_out')([shortcut, x])
    return x


def stack2(x, filters, blocks, stride1=2, name=None):
    """A set of stacked residual blocks.
    # Arguments
        x: input tensor.
        filters: integer, filters of the bottleneck layer in a block.
        blocks: integer, blocks in the stacked blocks.
        stride1: default 2, stride of the first layer in the first block.
        name: string, stack label.
    # Returns
        Output tensor for the stacked blocks.
    """
    x = block2(x, filters, conv_shortcut=True, name=name + '_block1')
    for i in range(2, blocks):
        x = block2(x, filters, name=name + '_block' + str(i))
    x = block2(x, filters, stride=stride1, name=name + '_block' + str(blocks))
    return x


def block3(x, filters, kernel_size=3, stride=1, groups=32,
           conv_shortcut=True, name=None):
    """A residual block.
    # Arguments
        x: input tensor.
        filters: integer, filters of the bottleneck layer.
        kernel_size: default 3, kernel size of the bottleneck layer.
        stride: default 1, stride of the first layer.
        groups: default 32, group size for grouped convolution.
        conv_shortcut: default True, use convolution shortcut if True,
            otherwise identity shortcut.
        name: string, block label.
    # Returns
        Output tensor for the residual block.
    """
    bn_axis = 3 if backend.image_data_format() == 'channels_last' else 1

    if conv_shortcut is True:
        shortcut = layers.Conv2D((64 // groups) * filters, 1, strides=stride,
                                 use_bias=False, name=name + '_0_conv')(x)
        shortcut = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                                             name=name + '_0_bn')(shortcut)
    else:
        shortcut = x

    x = layers.Conv2D(filters, 1, use_bias=False, name=name + '_1_conv')(x)
    x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                                  name=name + '_1_bn')(x)
    x = layers.Activation('relu', name=name + '_1_relu')(x)

    c = filters // groups
    x = layers.ZeroPadding2D(padding=((1, 1), (1, 1)), name=name + '_2_pad')(x)
    x = layers.DepthwiseConv2D(kernel_size, strides=stride, depth_multiplier=c,
                               use_bias=False, name=name + '_2_conv')(x)
    x_shape = backend.int_shape(x)[1:-1]
    x = layers.Reshape(x_shape + (groups, c, c))(x)
    output_shape = x_shape + (groups,
                              c) if backend.backend() == 'theano' else None

    x = layers.Lambda(lambda x: sum([x[:, :, :, :, i] for i in range(c)]),
                      output_shape=output_shape, name=name + '_2_reduce')(x)

    x = layers.Reshape(x_shape + (filters,))(x)

    x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                                  name=name + '_2_bn')(x)

    x = layers.Activation('relu', name=name + '_2_relu')(x)

    x = layers.Conv2D((64 // groups) * filters, 1, use_bias=False,
                      name=name + '_3_conv')(x)

    x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                                  name=name + '_3_bn')(x)

    x = layers.Add(name=name + '_add')([shortcut, x])
    x = layers.Activation('relu', name=name + '_out')(x)
    return x


def stack3(x, filters, blocks, stride1=2, groups=32, name=None):
    """A set of stacked residual blocks.
    # Arguments
        x: input tensor.
        filters: integer, filters of the bottleneck layer in a block.
        blocks: integer, blocks in the stacked blocks.
        stride1: default 2, stride of the first layer in the first block.
        groups: default 32, group size for grouped convolution.
        name: string, stack label.
    # Returns
        Output tensor for the stacked blocks.
    """
    x = block3(x, filters, stride=stride1, groups=groups,
               name=name + '_block1')

    for i in range(2, blocks + 1):
        x = block3(x, filters, groups=groups, conv_shortcut=False,
                   name=name + '_block' + str(i))
    return x
