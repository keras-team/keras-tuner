from tensorflow.keras.datasets import cifar10
from tensorflow.keras.optimizers import Adam

import os
import pickle
import numpy as np

#FIXME allows to swap to CIFAR 100
num_classes = 10

def residual(x, num_filters):
    residual = x    
    x = SeparableConv2D(num_filters, (3, 3), activation='selu', padding='same', kernel_initializer='lecun_normal', bias_initializer='zeros')(x)
    x = SeparableConv2D(num_filters, (3, 3), activation='selu', padding='same', kernel_initializer='lecun_normal', bias_initializer='zeros')(x)
#    x = AlphaDropout(0.1)(x)
    x = layers.add([x, residual])
    return x 

def conv(x, num_filters):
    x = SeparableConv2D(num_filters, (3, 3), activation='selu', padding='same', kernel_initializer='lecun_normal', bias_initializer='zeros')(x)
    return x


def model_fn():
    inputs = Input(shape=(32, 32, 3))
    x = inputs

    enc_

    for num_filters in range(encode_cnn_expand)
    x = conv(input_img, 8)
    x = conv(input_img, 16)
    #x = conv(input_img, 32)
    #x = conv(x, 64)
    
    for dims in [32]:
        x = conv(x, dims)
        x = residual(x, dims)
        #x = MaxPooling2D((3, 3), strides=(3,3))(x)

    #dims = dims * 2
    #x = conv(x, dims)
    for _ in range(2):
        x = residual(x, dims)
    

    
    x = Flatten()(x)    
    #x = GlobalAveragePooling2D()(x)
    #x = GlobalMaxPooling2D()(x)
    
    x = Dense(256,  activation='selu', kernel_initializer='lecun_normal', bias_initializer='zeros')(x)
    #decoder
    #x = AlphaDropout(0.1)(x)
    x = Dense(256,  activation='selu', kernel_initializer='lecun_normal', bias_initializer='zeros')(x)
    x = Reshape((16, 16, 1))(x)

    #decompress    
    for dims in [8, 16, 32]:
        x = conv(x, dims)
        #x = conv(x, dims)
    #for _ in range(4):
    #    x = residual(x, dims)
    x = residual(x, dims)
    #x = residual(x, dims)


    #reshape
    x = UpSampling2D((2, 2))(x)
    
    #reduce channels
    #arr = []
    #for dims in [128, 64, 32, 16, 8]: #, 16, 8, 3]:
    #    v = conv(x, dims)
    #    v = residual(v, dims)
    #     v = conv(v, 3) 
    #    arr.append(v)
    #x = conv(x, 3)
    #for _ in range(3):
    #    arr.append(residual(x, 3))
    ### alternative going deep
    #decoded = layers.add(arr)
    
    for dims in [32, 16, 8, 3]: #, 16, 8, 3]:
        x = conv(x, dims)
        #x = conv(x, dims)
    
    for _ in range(3):
        x = residual(x, dims)
    
    
    decoded = SeparableConv2D(3, (3, 3),  padding='same', activation='sigmoid')(x)
    return Model(input_img, decoded)







(x_train, y_train), (x_test, y_test) = cifar10.load_data()
# normalize data
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

model = model_fn()
model.compile(optimizer='adam', loss='binary_crossentropy')
model.summary()
history = model.fit(x_train, x_train, epochs=epochs, validation_split=0.1)
score = model.evaluate(x_test, x_test)
print("eval score:%s " % score)