"""Hypertunable version of Resnet."""

import functools

import tensorflow.keras as keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras import optimizers
from tensorflow.keras import backend

from kerastuner.applications.tunable_resnet import blocks

from kerastuner.applications.tunable_resnet.hparams import default_fixed_hparams, default_hparams


def resnet(input_shape, num_classes, **hparams):
    """ Returns a wrapper around a Resnet function which provides the
    specified shape, number of output classes, and hyperparameters.

    Args:
        input_shape (tuple):  Shape of the input image.
        num_classes (int): Number of output classes.
        hparams (**dictionary): Hyperparameters to the Resnet model.

    Returns:
        A wrapped resnet function.
    """

    return functools.partial(_resnet, input_shape, num_classes, **hparams)


def resnet_single_fn(input_shape, num_classes, **hparams):
    """ Returns a wrapper around a resnet_single function which provides the
    specified shape, number of output classes, and hyperparameters.

    Args:
        input_shape (tuple):  Shape of the input image.
        num_classes (int): Number of output classes.
        hparams (**dictionary): Hyperparameters to the Resnet model.

    Returns:
        A wrapped resnet_single function.
    """

    return functools.partial(
        _resnet_single,
        input_shape,
        num_classes,
        **hparams)


def resnet_single_model(input_shape, num_classes, **hparams):
    model_fn = resnet_single_fn(input_shape, num_classes, **hparams)
    return model_fn()


def _resnet(input_shape, num_classes, include_top=True, **hparams):
    """
    Implementation of a hypertunable adaptation of Resnet.
    """
    hp = {}
    hp.update(default_hparams(input_shape, num_classes))
    if hparams:
        hp.update(hparams)

    ### Parameters ###

    # [General]
    version = hp["version"]
    optimizer = hp["optimizer"]
    preact = hp["preact"]
    use_bias = hp["use_bias"]

    # [Variable stack]
    conv3_depth = hp["conv3_depth"]
    conv4_depth = hp["conv4_depth"]

    ### Model ###

    bn_axis = 3 if backend.image_data_format() == 'channels_last' else 1
    inputs = layers.Input(shape=input_shape)
    x = inputs

    # Initial conv2d block.
    x = layers.ZeroPadding2D(padding=((3, 3), (3, 3)), name='conv1_pad')(x)
    x = layers.Conv2D(
        64,
        7,
        strides=2,
        use_bias=use_bias,
        name='conv1_conv')(x)
    if preact is False:
        x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                                      name='conv1_bn')(x)
        x = layers.Activation('relu', name='conv1_relu')(x)
    x = layers.ZeroPadding2D(padding=((1, 1), (1, 1)), name='pool1_pad')(x)
    x = layers.MaxPooling2D(3, strides=2, name='pool1_pool')(x)

    # Middle hypertunable stack.
    if version == 'v1':
        x = blocks.stack1(x, 64, 3, stride1=1, name='conv2')
        x = blocks.stack1(x, 128, conv3_depth, name='conv3')
        x = blocks.stack1(x, 256, conv4_depth, name='conv4')
        x = blocks.stack1(x, 512, 3, name='conv5')
    elif version == 'v2':
        x = blocks.stack2(x, 64, 3, name='conv2')
        x = blocks.stack2(x, 128, conv3_depth, name='conv3')
        x = blocks.stack2(x, 256, conv4_depth, name='conv4')
        x = blocks.stack2(x, 512, 3, stride1=1, name='conv5')
    elif version == 'next':
        x = blocks.stack3(x, 64, 3, name='conv2')
        x = blocks.stack3(x, 256, conv3_depth, name='conv3')
        x = blocks.stack3(x, 512, conv4_depth, name='conv4')
        x = blocks.stack3(x, 1024, 3, stride1=1, name='conv5')

    # Top of the model.
    if preact is True:
        x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                                      name='post_bn')(x)
        x = layers.Activation('relu', name='post_relu')(x)

    if include_top:
        x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
        x = layers.Dense(num_classes, activation='softmax', name='probs')(x)
    else:
        if pooling == 'avg':
            x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
        elif pooling == 'max':
            x = layers.GlobalMaxPooling2D(name='max_pool')(x)

    model = keras.Model(inputs, x, name="Tunable Resnet")

    if optimizer == "adam":
        optimizer = optimizers.Adam(lr=hp["learning_rate"])
    elif optimizer == "sgd":
        optimizer = optimizers.SGD(
            lr=hp["learning_rate"],
            momentum=hp["momentum"],
            decay=hp["learning_rate_decay"])
    elif optimizer == "rmsprop":
        optimizer = optimizers.RMSprop(
            lr=hp["learning_rate"],
            decay=hp["learning_rate_decay"])
    else:
        raise ValueError("Optimizer '%s' not supported", optimizer)

    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy'])
    return model


def _resnet_single(
        input_shape,
        num_classes,
        **hparams):
    """ Model fn which uses a single value for the hyper parameters """

    hp = default_fixed_hparams(input_shape, num_classes)
    hp.update(hparams)
    return _resnet(input_shape=input_shape, num_classes=num_classes, **hp)
