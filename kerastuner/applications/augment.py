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


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.keras as keras

from kerastuner.engine import hypermodel

from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing

import random

# dict of functions that create layers for transforms.
# Each function takes a factor (0 to 1) for the strength
# of the transform.
TRANSFORMS = {
'translate_x': lambda x: preprocessing.RandomTranslation(x, 0),
'translate_y': lambda y: preprocessing.RandomTranslation(0, y),
'rotate': preprocessing.RandomRotation,
'contrast': preprocessing.RandomContrast,
}

class HyperAugment(hypermodel.HyperModel):
    """ Builds HyperModel for image augmentation.

    # Arguments:
        input_shape: Optional shape tuple, e.g. `(256, 256, 3)`.
        input_tensor: Optional Keras tensor (i.e. output of
            `layers.Input()`) to use as image input for the model.
        rotate: Whether to include rotation in augmentation.
        translate_x: Whether to include horizontal translation in augmentation.
        translate_y: Whether to include vertical translation in augmentation.
        contrast: Whether to include random contrast in augmentation.
        **kwargs: Additional keyword arguments that apply to all
            HyperModels. See `kerastuner.HyperModel`.
    """
    def __init__(self,
        input_shape=None,
        input_tensor=None,
        rotate=True,
        translate_x=True,
        translate_y=True,
        contrast=True):
        self.transforms = []
        if rotate:
            self.transforms.append('rotate')
        if translate_x:
            self.transforms.append('translate_x')
        if translate_y:
            self.transforms.append('translate_y')
        if contrast:
            self.transforms.append('contrast')

        self.input_shape = input_shape
        self.input_tensor = input_tensor

    def build(self, hp):
        raise NotImplemented

class HyperFixedAugment(HyperAugment):
    """An HyperModel for fixed policy augmentation.
       A tunable hyperparameter is assigned to each of the
       transforms chosen, and the transforms are applied
       sequentially.
       Only supporting augmentations in Keras preprocessing layers.

    # Arguments:
        input_shape: Optional shape tuple, e.g. `(256, 256, 3)`.
        input_tensor: Optional Keras tensor (i.e. output of
            `layers.Input()`) to use as image input for the model.
        rotate: Whether to include rotation in augmentation. When
             set to True, RandomRotation layer will be added to the
             pool of augmentations to select from.
        translate_x: Whether to include horizontal translation in augmentation.
        translate_y: Whether to include vertical translation in augmentation.
        contrast: Whether to include random contrast in augmentation.
        **kwargs: Additional keyword arguments that apply to all
            HyperModels. See `kerastuner.HyperModel`.

    # Search space:
        'factor_{transform}' for each of the transforms chosen:
            Float, default range [0.05, 1]. Controls the strength
            of the given transformation.
    """

    def build(self, hp):
        model = keras.Sequential(name='fixed_augment')

        if self.input_tensor is not None:
            model.add(self.input_tensor)
        elif self.input_shape is not None:
            model.add(keras.Input(shape=self.input_shape))

        for transform in self.transforms:
            transform_factor = hp.Float(f'factor_{transform}', 0.05, 1, step=0.05, default=0.15)
            transform_layer = TRANSFORMS[transform](transform_factor)
            model.add(transform_layer)

        return model

class HyperRandAugment(HyperAugment):
    """An HyperModel for Rand augmentation.
       Based on https://arxiv.org/abs/1909.13719.
       This augmentation randomly picks `randaug_count` augmentation transforms
       from a pool of transforms for each sample, and set the augmentation
       magnitude to `randaug_mag`.
       Only supporting augmentations in Keras preprocessing layers.

    # Arguments:
        input_shape: Optional shape tuple, e.g. `(256, 256, 3)`.
              One of `input_shape` or `input_tensor` must be
              specified.
        input_tensor: Optional Keras tensor (i.e. output of
            `layers.Input()`) to use as image input for the model.
              One of `input_shape` or `input_tensor` must be
              specified.
        rotate: Whether to include rotation in augmentation. When
             set to True, RandomRotation layer will be added to the
             pool of augmentations to select from.
        translate_x: Whether to include horizontal translation in augmentation.
        translate_y: Whether to include vertical translation in augmentation.
        contrast: Whether to include random contrast in augmentation.
        **kwargs: Additional keyword arguments that apply to all
            HyperModels. See `kerastuner.HyperModel`.

    # Search space:
        'randaug_mag': Float, default range [0.05, 1].
                       Controls the strength of each augmentation
                       applied to each image.
        'randaug_count': Int, default range [0, 5].
                       Controls the number of augmentations layers
                       applied to each image.
    """
    def __init__(self,
                 input_shape=None,
                 input_tensor=None,
                 **kwargs):
        super(HyperRandAugment, self).__init__(input_shape=input_shape,
            input_tensor=input_tensor,
            **kwargs)
        if input_shape is None and input_tensor is None:
            raise ValueError('You must specify either `input_shape` or '
                             '`intput_tensor` when using HyperRandAugment.')

    def build(self, hp):
        if self.input_tensor is not None:
            inputs = keras.utils.get_source_inputs(self.input_tensor)
            x = self.input_tensor
        else:
            inputs = layers.Input(shape=self.input_shape)
            x = inputs

        randaug_mag = hp.Float('randaug_mag', 0.05, 1.0, step=0.1, default=0.2)
        randaug_count = hp.Int('randaug_count', 0, 5, default=2)

        for _ in range(randaug_count):
            # selection tensor determines for each sample, which operation
            # is used.
            batch_size = tf.shape(x)[0]
            selection = tf.random.uniform([batch_size, 1, 1, 1],
                maxval=len(self.transforms),
                dtype='int32')

            for (i, transform) in enumerate(self.transforms):
                transform_layer = TRANSFORMS[transform](randaug_mag)
                x_trans = transform_layer(x)

                # for each sample, apply the transform if and only if
                # selection matches the transform index `i`
                x = tf.where(tf.equal(i, selection), x_trans, x)
                
        model = keras.Model(inputs, x, name='rand_augment')
        return model

