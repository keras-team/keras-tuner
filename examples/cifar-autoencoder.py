from tensorflow.keras.datasets import cifar100, cifar10
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.callbacks import  LearningRateScheduler

import tensorflow as tf
from math import exp

import os
import pickle
import numpy as np

#FIXME allows to swap to CIFAR 100
dataset = 'cifar10'

epochs = 100
#config = tf.ConfigProto(device_count = {'GPU': 0})
#sess = tf.Session(config=config)

def exp_decay(epoch):
   initial_lrate = 0.001
   k = 0.1
   lrate = initial_lrate * exp(-k * epoch)
   return lrate
lrate = LearningRateScheduler(exp_decay)

def sep_conv(x, num_filters, kernel_size=(3, 3), activation='relu'):
    if activation == 'selu':
        x = layers.SeparableConv2D(num_filters, kernel_size, activation='selu', padding='same', kernel_initializer='lecun_normal', bias_initializer='zeros')(x)
    elif activation == 'relu':
        x = layers.SeparableConv2D(num_filters, kernel_size, padding='same', use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
    else:
        msg = 'Unkown activation function: %s' % activation
        ValueError(msg)
    return x

def residual(x, num_filters, kernel_size=(3, 3), activation='relu', max_pooling=False):
    "Residual block"
    residual = x    
    x = sep_conv(x, num_filters, kernel_size, activation)
    x = sep_conv(x, num_filters, kernel_size, activation)
    x = layers.add([x, residual])
    if max_pooling:
        x = layers.MaxPooling2D(kernel_size, strides=(2, 2), padding='same')(x)
    return x 

def conv(x, num_filters, kernel_size=(3, 3), activation='relu', strides=(2, 2)):
    "2d convolution block"
    if activation == 'selu':
        x = layers.Conv2D(num_filters, kernel_size, strides=strides, activation='selu', padding='same', kernel_initializer='lecun_normal', bias_initializer='zeros')(x)

    elif activation == 'relu':
        x = layers.Conv2D(num_filters, kernel_size, strides=strides, use_bias=False, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
    else:
        msg = 'Unkown activation function: %s' % activation
        ValueError(msg)
    return x
    
def dense(x, dims, activation='relu', batchnorm=True, dropout_rate=0):
    if activation == 'selu':
        x = layers.Dense(dims,  activation='selu', kernel_initializer='lecun_normal', bias_initializer='zeros')(x)
    elif activation  == 'relu':
        x = layers.Dense(dims, activation='relu')(x)
        if batchnorm:
            x = layers.BatchNormalization()(x)
        if dropout_rate:
            x = layers.Dropout(dropout_rate)(x)
    else:
        msg = 'Unkown activation function: %s' % activation
        ValueError(msg)
    return x 

def model_fn():
    inputs = layers.Input(shape=(32, 32, 3))
    x = inputs

    kernel_size = (3, 3)
    activation = 'selu'

    enc_expand_start_dims = 32
    enc_expand_stop_dims = 256
    enc_num_residual_blocks = 3

    dense_merge_type = 'flatten'
    dense_outer_dims = 512
    dense_inner_dims = 256
    dense_use_bn = True

    dec_num_residual_blocks = 6

    #inflate filters
    dims = enc_expand_start_dims
    while dims <= enc_expand_stop_dims:
        x = conv(x, dims, activation=activation)
        x = residual(x, dims, activation=activation)
        dims *= 2
        
    x = conv(x, dims, activation=activation)
    
    #residual blocks
    for _ in range(enc_num_residual_blocks):
        x = residual(x, dims, activation=activation)

    if dense_merge_type == 'flatten':
        x = layers.Flatten()(x)
    elif dense_merge_type == "avg":
        x = layers.GlobalAveragePooling2D()(x)
    else:
        x = layers.GlobalMaxPooling2D()(x)
    
    #compress
    dense_dims = dense_outer_dims
    while dense_dims > dense_inner_dims:
        x = dense(x, dense_dims, activation=activation, batchnorm=dense_use_bn)
        dense_dims /= 2
    
    x = dense(x, dense_inner_dims, activation=activation, batchnorm=dense_use_bn)    
    

    #decoder
    ## decompress
    dense_dims = dense_inner_dims *2
    while dense_dims/2 < 1024:
        x = dense(x, dense_dims, activation=activation, batchnorm=dense_use_bn)
        dense_dims *= 2
    
    x = dense(x, 3072, activation=activation, batchnorm=dense_use_bn)
    x = layers.Reshape((32, 32, 3))(x)
    #print(x.shape)

    for _ in range(dec_num_residual_blocks):
        x = residual(x, 3, activation=activation)

    
    
    decoded = layers.SeparableConv2D(3, (3, 3),  padding='same', activation='sigmoid')(x)
    return Model(inputs, decoded)






if dataset == 'cifar100':
    (x_train, y_train), (x_test, y_test) = cifar100.load_data()
else:
   (x_train, y_train), (x_test, y_test) = cifar10.load_data() 
# normalize data
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

model = model_fn()
model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['mae'])
model.summary()
history = model.fit(x_train, x_train, epochs=epochs, batch_size=32,  callbacks=[lrate], validation_data=(x_test, x_test))
model.save('cifar-auto-encode.mdl')
score = model.evaluate(x_test, x_test)
print("eval score:%s " % score)