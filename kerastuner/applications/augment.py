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
    Only supporting augmentations available in Keras preprocessing layers currently.

    # Arguments:
        input_shape: Optional shape tuple, e.g. `(256, 256, 3)`.
        input_tensor: Optional Keras tensor (i.e. output of
            `layers.Input()`) to use as image input for the model.
        rotate: A number between [0, 1], a list of two numbers between [0, 1]
            or None. Configures the search space of the factor of random
            rotation transform in the augmentation. A factor is chosen for each
            trial. It sets maximum of clockwise and counterclockwise rotation
            in terms of fraction of pi, among all samples in the trial.
            Default is 0.5. When `rotate` is a single number, the search range is
            [0, `rotate`].
            The transform is off when set to None.
        translate_x: A number between [0, 1], a list of two numbers between [0, 1]
            or None. Configures the search space of the factor of random
            horizontal translation transform in the augmentation. A factor is
            chosen for each trial. It sets maximum of horizontal translation in
            terms of ratio over the width among all samples in the trial.
            Default is 0.4. When `translate_x` is a single number, the search range
            is [0, `translate_x`].
            The transform is off when set to None.
        translate_y: A number between [0, 1], a list of two numbers between [0, 1]
            or None. Configures the search space of the factor of random vertical
            translation transform in the augmentation. A factor is chosen for each
            trial. It sets maximum of vertical translation in terms of ratio over
            the height among all samples in the trial. Default is 0.4. When
            `translate_y` is a single number ,the search range is [0, `translate_y`].
            The transform is off when set to None.
        contrast: A number between [0, 1], a list of two numbers between [0, 1]
            or None. Configures the search space of the factor of random contrast
            transform in the augmentation. A factor is chosen for each trial. It
            sets maximum ratio of contrast change among all samples in the trial.
            Default is 0.3. When `contrast` is a single number, the search rnage is
            [0, `contrast`].
            The transform is off when set to None.
        augment_layers: None, int or list of two ints, controlling the number
            of augment applied. Default is 3.
            When `augment_layers` is 0, all transform are applied sequentially.
            When `augment_layers` is nonzero, or a list of two ints, a simple
            version of RandAugment(https://arxiv.org/abs/1909.13719) is used.
            A search space for 'augment_layers' is created to search [0,
            `augment_layers`], or between the two ints if a `augment_layers` is
            a list. For each trial, the hyperparameter 'augment_layers'
            determines number of layers of augment transforms are applied,
            each randomly picked from all available transform types with equal
            probability on each sample.
        **kwargs: Additional keyword arguments that apply to all
            HyperModels. See `kerastuner.HyperModel`.

    # Search space:
        'factor_{transform}' for each of the transforms chosen:
            Float, range is set through corresponding keyword argument.
        'augment_layers' for number of transform applied to each sample:
            Int, range is set through keyword argument `augment_layers`.
            This is only active when `augment_layers` is non zero.

    # Usage example:
        ```
        hm_aug = HyperImageAugment(input_shape=(32, 32, 3),
                                   augment_layers=0,
                                   rotate=[0.2, 0.3],
                                   translate_x=0.1,
                                   translate_y=None,
                                   contrast=None)
        ```
        Then the hypermodel `hm_aug` will search 'factor_rotate' between [0.2, 0.3]
        and 'factor_translate_x' between [0, 0.1]. These two augments are applied
        on all samples with factor picked per each trial.
        ```
        hm_aug = HyperImageAugment(input_shape=(32, 32, 3),
                                   translate_x=0.5,
                                   translate_y=[0.2, 0.4]
                                   contrast=None)
        ```
        Then the hypermodel `hm_aug` will search 'factor_rotate' between [0, 0.2],
        'factor_translate_x' between [0, 0.5], 'factor_translate_y' between
        [0.2, 0.4]. It will use RandAugment, searching 'augment_layers'
        between [0, 3]. Each layer on each sample will be chosen from rotate,
        translate_x and translate_y.
    """
    def __init__(self,
                 input_shape=None,
                 input_tensor=None,
                 rotate=0.5,
                 translate_x=0.4,
                 translate_y=0.4,
                 contrast=0.3,
                 augment_layers=3,
                 **kwargs):

        if input_shape is None and input_tensor is None:
            raise ValueError('You must specify either `input_shape` or '
                             '`intput_tensor`.')

        self.transforms = []
        self._register_transform('rotate', rotate)
        self._register_transform('translate_x', translate_x)
        self._register_transform('translate_y', translate_y)
        self._register_transform('contrast', contrast)

        self.input_shape = input_shape
        self.input_tensor = input_tensor

        if augment_layers:
            self.model_name = 'image_rand_augment'
            try:
                augment_layers_min = augment_layers[0]
                augment_layers_max = augment_layers[1]
            except TypeError:
                augment_layers_min = 0
                augment_layers_max = augment_layers
            if not (isinstance(augment_layers_min, int) and
                    isinstance(augment_layers_max, int)):
                raise ValueError('Keyword argument `augment_layers` must be int,'
                                 'but received {}. '.format(augment_layers))

            self.augment_layers_min = augment_layers_min
            self.augment_layers_max = augment_layers_max
        else:
            # Separatedly tune and apply all augment transforms if
            # `randaug_count` is set to 0.
            self.model_name = 'image_augment'

        super(HyperImageAugment, self).__init__(**kwargs)

    def build(self, hp):
        if self.input_tensor is not None:
            inputs = keras.utils.get_source_inputs(self.input_tensor)
            x = self.input_tensor
        else:
            inputs = layers.Input(shape=self.input_shape)
            x = inputs

        if self.model_name == 'image_rand_augment':
            x = self._build_randaug_layers(x, hp)
        else:
            x = self._build_fixedaug_layers(x, hp)

        model = keras.Model(inputs, x, name=self.model_name)
        return model

    def _build_randaug_layers(self, inputs, hp):
        augment_layers = hp.Int('augment_layers',
                                self.augment_layers_min,
                                self.augment_layers_max,
                                default=self.augment_layers_min)
        x = inputs
        for _ in range(augment_layers):
            # selection tensor determines operation for each sample.
            batch_size = tf.shape(x)[0]
            selection = tf.random.uniform([batch_size, 1, 1, 1],
                                          maxval=len(self.transforms),
                                          dtype='int32')

            for i, (transform, (f_min, f_max)) in enumerate(self.transforms):
                # Factor for each transform is determined per each trial.
                factor = hp.Float(f'factor_{transform}',
                                  f_min,
                                  f_max,
                                  default=f_min)
                if factor == 0:
                    continue
                transform_layer = TRANSFORMS[transform](factor)
                x_trans = transform_layer(x)

                # For each sample, apply the transform if and only if
                # selection matches the transform index `i`
                x = tf.where(tf.equal(i, selection), x_trans, x)
        return x

    def _build_fixedaug_layers(self, inputs, hp):
        x = inputs
        for transform, (factor_min, factor_max) in self.transforms:
            transform_factor = hp.Float(f'factor_{transform}',
                                        factor_min,
                                        factor_max,
                                        step=0.05,
                                        default=factor_min)
            if transform_factor == 0:
                continue
            transform_layer = TRANSFORMS[transform](transform_factor)
            x = transform_layer(x)
        return x

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
