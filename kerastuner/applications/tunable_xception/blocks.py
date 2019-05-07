import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers
from tensorflow.keras.layers import *


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


def conv(x, num_filters, kernel_size=(3, 3), activation='relu', strides=(2, 2)):
    "2d convolution block."
    if activation == 'selu':
        x = layers.Conv2D(num_filters, kernel_size, strides=strides, activation='selu',
                          padding='same', kernel_initializer='lecun_normal', bias_initializer='zeros')(x)
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
                         kernel_initializer='lecun_normal', bias_initializer='zeros')(x)
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
