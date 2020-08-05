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
        augmentation_model: optional Model or HyperModel for image augmentation.
        **kwargs: Additional keyword arguments that apply to all
            HyperModels. See `kerastuner.HyperModel`.
    """
    def __init__(self,
                 input_shape=None,
                 input_tensor=None,
                 classes=None,
                 augmentation_model=None,
                 **kwargs):
        if not isinstance(augmentation_model, (hypermodel.HyperModel,
                                               keras.Model,
                                               type(None))):
            raise ValueError('Keyword augmentation_model should be '
                             'a HyperModel, a Keras Model or empty. '
                             'Received {}.'.format(augmentation_model))

        if not classes:
            raise ValueError('You must specify `classes`')

        if input_shape is None and input_tensor is None:
            raise ValueError('You must specify either `input_shape` '
                             'or `input_tensor`.')

        self.input_shape = input_shape
        self.input_tensor = input_tensor
        self.classes = classes
        self.augmentation_model = augmentation_model

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

        x = preprocessing.Resizing(img_size, img_size, interpolation='bilinear')(x)
        efficientnet_model = EFFICIENTNET_MODELS[version](include_top=False,
                                                          input_tensor=x)

        # Rebuild top layers of the model.
        x = efficientnet_model.output

        pooling = hp.Choice('pooling', ['avg', 'max'], default='avg')
        if pooling == 'avg':
            x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
        elif pooling == 'max':
            x = layers.GlobalMaxPooling2D(name='max_pool')(x)

        top_dropout_rate = hp.Float('top_dropout_rate',
                                    min_value=0.2,
                                    max_value=0.8,
                                    step=0.2,
                                    default=0.2)
        x = layers.Dropout(top_dropout_rate, name='top_dropout')(x)

        x = layers.Dense(
            self.classes, activation='softmax', name='probs')(x)

        # compile
        model = keras.Model(inputs, x, name='EfficientNet')
        self._compile(model, hp)

        return model

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
