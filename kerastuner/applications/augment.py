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

# dict of functions that create layers for transforms.
# Each function takes a factor (0 to 1) for the strength
# of the transform.
TRANSFORMS = {
    'translate_x': lambda x: preprocessing.RandomTranslation(x, 0),
    'translate_y': lambda y: preprocessing.RandomTranslation(0, y),
    'rotate': preprocessing.RandomRotation,
    'contrast': preprocessing.RandomContrast,
}


class HyperImageAugment(hypermodel.HyperModel):
    """ Builds HyperModel for image augmentation.

    # Arguments:
        input_shape: Optional shape tuple, e.g. `(256, 256, 3)`.
        input_tensor: Optional Keras tensor (i.e. output of
            `layers.Input()`) to use as image input for the model.
        rotate: A number between [0, 1], a list of two numbers between [0, 1]
            or None. It configures the range of factor of rotation
            transform in the augmentation. Default is 0.2.
        translate_x: A number between [0, 1], a list of two numbers between [0, 1]
            or None. It configures the range of factor of horizontal translation
            transform in the augmentation. Default is 0.1.
        translate_y: A number between [0, 1], a list of two numbers between [0, 1]
            or None. It configures the range of factor of vertical translation
            transform in the augmentation. Default is 0.1.
        contrast: A number between [0, 1], a list of two numbers between [0, 1]
            or None. It configures the range of factor of contrast
            transform in the augmentation. Default is 0.1.
        **kwargs: Additional keyword arguments that apply to all
            HyperModels. See `kerastuner.HyperModel`.

    # Notes for keyword args that configures range of factor of transform:
        rotate, translate_x, translate_y and contrast:
            If two numbers are given, they represent minimum and maximum
            factors; if one number is given, it represents maximum factor, and
            minimum factor is 0; if None, the transform is disabled. All factors
            are within [0, 1].
    """
    def __init__(self,
                 input_shape=None,
                 input_tensor=None,
                 rotate=0.2,
                 translate_x=0.1,
                 translate_y=0.1,
                 contrast=0.1,
                 **kwargs):

        self.transforms = []
        self._register_transform('rotate', rotate)
        self._register_transform('translate_x', translate_x)
        self._register_transform('translate_y', translate_y)
        self._register_transform('contrast', contrast)

        self.input_shape = input_shape
        self.input_tensor = input_tensor
        super(HyperImageAugment, self).__init__(**kwargs)

    def build(self, hp):
        raise NotImplementedError

    def _register_transform(self, transform_name, transform_params):
        """Register a transform and format parameters for tuning the transform.
        # Arguments:
            transform_name: str, the name of the transform.
            trnasform_params: A number between [0, 1], a list of two numbers
                between [0, 1] or None. If set to a single number x, the
                corresponding transform factor will be between [0, x].
                If set to a list of 2 numbers [x, y], the factor will be
                between [x, y]. If set to None, the transform will be excluded.
        """
        if not transform_params:
            return

        try:
            transform_factor_min = transform_params[0]
            transform_factor_max = transform_params[1]
            if len(transform_params) > 2:
                raise ValueError('Length of keyword argument {} must not exceed 2.'
                                 .format(transform_name))
        except TypeError:
            transform_factor_min = 0
            transform_factor_max = transform_params

        if not (isinstance(transform_factor_max, (int, float)) and
                isinstance(transform_factor_min, (int, float))):
            raise ValueError('Keyword argument {} must be int or float, '
                             'but received {}. '
                             .format(transform_name, transform_params))

        self.transforms.append((transform_name,
                                (transform_factor_min, transform_factor_max)))


class HyperImageFixedAugment(HyperImageAugment):
    """An HyperModel for fixed policy augmentation.
       A tunable hyperparameter is assigned to each of the
       transforms chosen, and the transforms are applied
       sequentially.
       Only supporting augmentations in Keras preprocessing layers.

    # Arguments:
        input_shape: Optional shape tuple, e.g. `(256, 256, 3)`.
        input_tensor: Optional Keras tensor (i.e. output of
            `layers.Input()`) to use as image input for the model.
        **kwargs: Additional keyword arguments that apply to all
            HyperImageAugment. See `HyperImageAugment`.

    # Search space:
        'factor_{transform}' for each of the transforms chosen:
            Float, range is set through corresponding keyword argument.

    # Usage example:
        ```
        hm_aug = HyperImageFixedAugment(input_shape=(32, 32, 3),
                                        translate_x=0.5,
                                        translate_y=[0.2, 0.4]
                                        contrast=0)
        ```
        Then the hypermodel `hm_aug` will search 'factor_rotate' in [0, 0.2],
        'factor_translate_x' in [0, 0.5], 'factor_translate_y' in [0.2, 0.4].
    """

    def build(self, hp):
        model = keras.Sequential(name='fixed_augment')

        if self.input_tensor is not None:
            model.add(self.input_tensor)
        elif self.input_shape is not None:
            model.add(keras.Input(shape=self.input_shape))

        for transform, (factor_min, factor_max) in self.transforms:
            transform_factor = hp.Float(f'factor_{transform}',
                                        factor_min,
                                        factor_max,
                                        step=0.05,
                                        default=factor_min)
            if transform_factor == 0:
                continue
            transform_layer = TRANSFORMS[transform](transform_factor)
            model.add(transform_layer)

        return model


class HyperImageRandAugment(HyperImageAugment):
    """An HyperModel for RandAugment for image.
       Based on https://arxiv.org/abs/1909.13719.
       This augmentation randomly picks `randaug_count` augmentation transforms
       from a pool of transforms for each sample, and control the augmentation
       magnitude of all transforms using one parameter `randaug_mag`.
       Only supporting augmentations in Keras preprocessing layers, which is a
       subset of the larger pool of augmentation transforms in the original
       paper.

    # Arguments:
        input_shape: Optional shape tuple, e.g. `(256, 256, 3)`.
            One of `input_shape` or `input_tensor` must be
            specified.
        input_tensor: Optional Keras tensor (i.e. output of
            `layers.Input()`) to use as image input for the model.
            One of `input_shape` or `input_tensor` must be
            specified.
        randaug_mag: Number or list of 2 numbers. Default is 1. It
            configures search space 'randaug_mag'. When there is one number x,
            this hyperparameter is tuned betwee [0, x]; when there are two
            numbers x and y, this hyperparameter is tuned between [x, y].
        randaug_count: int or list of 2 int. Default is 4.  It
            configures search space 'randaug_count'. When there is one number
            x, this hyperparameter is tuned betwee [0, x]; when there are two
            numbers x and y, this hyperparameter is tuned between [x, y].
        **kwargs: Additional keyword arguments that apply to all
            HyperImageAugment. See `HyperImageAugment`.

    # Search space:
        'randaug_mag': Float, set through `randaug_mag` keyword argument. Always
            within [0, 1]. Controls the strength of each augmentation applied
            to each image. For a transform with factor range set to [a, b],
            factor in a trial will be a + (b - a) * randaug_mag.
        'randaug_count': Int, set through `randaug_count` keyword argument. Always
            nonnegative. Controls the number of augmentations layers applied to
            each image.

    # Usage example:
        ```
        hm_aug = HyperImageRandAugment(input_shape=(32, 32, 3),
                                       translate_x=0.5,
                                       translate_y=[0.2, 0.4]
                                       contrast=0,
                                       randaug_mag=[0.4, 0.6])
        ```
        Then the hypermodel `hm_aug` will search 'randaug_mag' in [0.2, 0.4],
        'randaug_count' in [0, 5]. Contrast transform is turned off.
        In a trial with randaug_mag = 0.5, we have factors for rotate,
        translate_x, translate_y at 0.1, 0.25, 0.3 respectively.
    """
    def __init__(self,
                 input_shape=None,
                 input_tensor=None,
                 randaug_mag=1,
                 randaug_count=4,
                 **kwargs):
        if input_shape is None and input_tensor is None:
            raise ValueError('You must specify either `input_shape` or '
                             '`intput_tensor` when using HyperImageRandAugment.')

        try:
            randaug_mag_min = randaug_mag[0]
            randaug_mag_max = randaug_mag[1]
        except TypeError:
            randaug_mag_min = 0
            randaug_mag_max = randaug_mag
        if not (isinstance(randaug_mag_min, (int, float)) and
                isinstance(randaug_mag_max, (int, float))):
            raise ValueError('Keyword argument `randaug_mag` must be float or int,'
                             'but received {}. '.format(randaug_mag))

        try:
            randaug_count_min = randaug_count[0]
            randaug_count_max = randaug_count[1]
        except TypeError:
            randaug_count_min = 0
            randaug_count_max = randaug_count
        if not (isinstance(randaug_count_min, int) and
                isinstance(randaug_count_max, int)):
            raise ValueError('Keyword argument `randaug_count` must be int,'
                             'but received {}. '.format(randaug_mag))

        self.randaug_mag_min = randaug_mag_min
        self.randaug_mag_max = randaug_mag_max
        self.randaug_count_min = randaug_count_min
        self.randaug_count_max = randaug_count_max

        super(HyperImageRandAugment, self).__init__(input_shape=input_shape,
                                                    input_tensor=input_tensor,
                                                    **kwargs)

    def build(self, hp):
        if self.input_tensor is not None:
            inputs = keras.utils.get_source_inputs(self.input_tensor)
            x = self.input_tensor
        else:
            inputs = layers.Input(shape=self.input_shape)
            x = inputs

        randaug_mag = hp.Float('randaug_mag',
                               self.randaug_mag_min,
                               self.randaug_mag_max,
                               step=0.1,
                               default=self.randaug_mag_min)
        randaug_count = hp.Int('randaug_count',
                               self.randaug_count_min,
                               self.randaug_count_max,
                               default=self.randaug_count_min)

        for _ in range(randaug_count):
            # selection tensor determines operation for each sample.
            batch_size = tf.shape(x)[0]
            selection = tf.random.uniform([batch_size, 1, 1, 1],
                                          maxval=len(self.transforms),
                                          dtype='int32')

            for i, (transform, (f_min, f_max)) in enumerate(self.transforms):
                factor = f_min + (f_max - f_min) * randaug_mag
                if factor == 0:
                    continue
                transform_layer = TRANSFORMS[transform](factor)
                x_trans = transform_layer(x)

                # for each sample, apply the transform if and only if
                # selection matches the transform index `i`
                x = tf.where(tf.equal(i, selection), x_trans, x)

        model = keras.Model(inputs, x, name='rand_augment')
        return model
