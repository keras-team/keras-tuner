# Copyright 2020 Google LLC. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Hypertunable version of EfficientNet based on Keras.applications."""

import tensorflow as tf
import tensorflow.keras as keras

from tensorflow.keras import layers
from tensorflow.keras.applications import efficientnet
from tensorflow.keras.layers.experimental import preprocessing

from kerastuner.engine import hypermodel

import os


EFFICIENTNET_MODELS = {'B0': efficientnet.EfficientNetB0,
                       'B1': efficientnet.EfficientNetB1,
                       'B2': efficientnet.EfficientNetB2,
                       'B3': efficientnet.EfficientNetB3,
                       'B4': efficientnet.EfficientNetB4,
                       'B5': efficientnet.EfficientNetB5,
                       'B6': efficientnet.EfficientNetB6,
                       'B7': efficientnet.EfficientNetB7}

EFFICIENTNET_IMG_SIZE = {'B0': 224,
                         'B1': 240,
                         'B2': 260,
                         'B3': 300,
                         'B4': 380,
                         'B5': 456,
                         'B6': 528,
                         'B7': 600}


class HyperEfficientNet(hypermodel.HyperModel):
    """An EfficientNet HyperModel.
    Models built by this HyperModel takes input image data in
    ints [0, 255]. The output data should be one-hot encoded
    with number of classes matching `classes`.

      # Arguments:

        include_top: whether to include the fully-connected
            layer at the top of the network. Model is not
            compiled if include_top is set to False.
        input_shape: shape tuple, e.g. `(256, 256, 3)`.
              Input images will be resized if different from
              the default input size of the version of
              efficientnet base model used.
              One of `input_shape` or `input_tensor` must be
              specified.
        input_tensor: Keras tensor to use as image input for the model.
              One of `input_shape` or `input_tensor` must be
              specified.
        classes: number of classes to classify images into.
        weights: str or None. Default is 'imagenet', where the weights pre-trained
              on imagenet will be downloaded. Otherwise the weights will be
              loaded from the directory in 'weights', and are expected to be in
              h5 format with naming convention '{weights}/b{n}_notop.h5' where n
              is 0 to 7. If set to None, the weights will be initiated from scratch.
        augmentation_model: optional Model or HyperModel for image augmentation.
        **kwargs: Additional keyword arguments that apply to all
            HyperModels. See `kerastuner.HyperModel`.
    """
    def __init__(self,
                 include_top=True,
                 input_shape=None,
                 input_tensor=None,
                 classes=None,
                 weights='imagenet',
                 augmentation_model=None,
                 **kwargs):
        if not isinstance(augmentation_model, (hypermodel.HyperModel,
                                               keras.Model,
                                               type(None))):
            raise ValueError('Keyword augmentation_model should be '
                             'a HyperModel, a Keras Model or empty. '
                             'Received {}.'.format(augmentation_model))

        if include_top and classes is None:
            raise ValueError('You must specify `classes` when '
                             '`include_top=True`')

        if input_shape is None and input_tensor is None:
            raise ValueError('You must specify either `input_shape` '
                             'or `input_tensor`.')

        self.include_top = include_top
        self.input_shape = input_shape
        self.input_tensor = input_tensor
        self.classes = classes
        self.augmentation_model = augmentation_model
        self.weights = weights

        super(HyperEfficientNet, self).__init__(**kwargs)

    def build(self, hp):

        if self.input_tensor is not None:
            inputs = tf.keras.utils.get_source_inputs(self.input_tensor)
            x = self.input_tensor
        else:
            inputs = layers.Input(shape=self.input_shape)
            x = inputs

        if self.augmentation_model:
            if isinstance(self.augmentation_model, hypermodel.HyperModel):
                augmentation_model = self.augmentation_model.build(hp)
            elif isinstance(self.augmentation_model, keras.models.Model):
                augmentation_model = self.augmentation_model

            x = augmentation_model(x)

        # Select one of pre-trained EfficientNet as feature extractor
        version = hp.Choice('version',
                            ['B{}'.format(i) for i in range(8)],
                            default='B0')
        img_size = EFFICIENTNET_IMG_SIZE[version]

        weights = self.weights
        if weights and (weights != 'imagenet'):
            weights = os.path.join(weights, version.lower())
            weights += '_notop.h5'
            if not os.path.isfile(weights):
                raise ValueError('Expect path {} to include weight file; but '
                                 'no file is found'.format(weights))

        x = preprocessing.Resizing(img_size, img_size, interpolation='bilinear')(x)
        efficientnet_model = EFFICIENTNET_MODELS[version](include_top=False,
                                                          input_tensor=x,
                                                          weights=weights)

        # Rebuild top layers of the model.
        x = efficientnet_model.output

        pooling = hp.Choice('pooling', ['avg', 'max'], default='avg')
        if pooling == 'avg':
            x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
        elif pooling == 'max':
            x = layers.GlobalMaxPooling2D(name='max_pool')(x)

        if self.include_top:
            top_dropout_rate = hp.Float('top_dropout_rate',
                                        min_value=0.2,
                                        max_value=0.8,
                                        default=0.2)
            x = layers.Dropout(top_dropout_rate, name='top_dropout')(x)
            if self.classes == 1:
                x = layers.Dense(1, activation='sigmoid', name='probs')(x)
            else:
                x = layers.Dense(self.classes, activation='softmax', name='probs')(x)

            # compile
            model = keras.Model(inputs, x, name='EfficientNet')
            self._compile(model, hp)

            return model
        else:
            return keras.Model(inputs, x, name='EfficientNet')

    def _compile(self, model, hp):
        """ Compile model using hyperparameters in hp.
            When subclassing the hypermodel, this may
            be overriden to change behavior of compiling.
        """
        learning_rate = hp.Choice('learning_rate', [0.1, 0.01, 0.001], default=0.01)
        optimizer = tf.keras.optimizers.SGD(
                momentum=0.1,
                learning_rate=learning_rate)

        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy'])
