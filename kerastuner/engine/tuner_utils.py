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
"""Utilities for Tuner class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import numpy as np
import six

import tensorflow as tf
from tensorflow import keras

from ..abstractions import display


class TunerStats(object):
    """Track tuner statistics."""

    def __init__(self):
        self.num_generated_models = 0  # overall number of instances generated
        self.num_invalid_models = 0  # how many models didn't work
        self.num_oversized_models = 0  # num models with params> max_params

    def summary(self, extended=False):
        display.subsection('Tuning stats')
        display.display_settings(self.get_config())

    def get_config(self):
        return {
            'num_generated_models': self.num_generated_models,
            'num_invalid_models': self.num_invalid_models,
            'num_oversized_models': self.num_oversized_models
        }

    @classmethod
    def from_config(cls, config):
        stats = cls()
        stats.num_generated_models = config['num_generated_models']
        stats.num_invalid_models = config['num_invalid_models']
        stats.num_oversized_models = config['num_oversized_models']
        return stats


def get_max_epochs_and_steps(fit_args, fit_kwargs):
    if fit_args:
        x = tf.nest.flatten(fit_args)[0]
    else:
        x = tf.nest.flatten(fit_kwargs.get('x'))[0]
    batch_size = fit_kwargs.get('batch_size', 32)
    if hasattr(x, '__len__'):
        max_steps = math.ceil(float(len(x)) / batch_size)
    else:
        max_steps = fit_kwargs.get('steps')
    max_epochs = fit_kwargs.get('epochs', 1)
    return max_epochs, max_steps


class TunerCallback(keras.callbacks.Callback):

    def __init__(self, tuner, trial):
        super(TunerCallback, self).__init__()
        self.tuner = tuner
        self.trial = trial

    def on_epoch_begin(self, epoch, logs=None):
        self.tuner.on_epoch_begin(
            self.trial, self.model, epoch, logs=logs)

    def on_batch_begin(self, batch, logs=None):
        self.tuner.on_batch_begin(self.trial, self.model, batch, logs)

    def on_batch_end(self, batch, logs=None):
        self.tuner.on_batch_end(self.trial, self.model, batch, logs)

    def on_epoch_end(self, epoch, logs=None):
        self.tuner.on_epoch_end(
            self.trial, self.model, epoch, logs=logs)


# TODO: Add more extensive display.
class Display(object):

    def __init__(self, verbose=1):
        self.verbose = verbose

    def on_trial_begin(self, trial):
        if self.verbose >= 1:
            display.section('Starting new trial')

    def on_trial_end(self, trial):
        if self.verbose >= 1:
            display.section('Trial complete')
            trial.summary()


def average_histories(histories):
    """Averages the per-epoch metrics from multiple executions."""
    averaged = {}
    metrics = histories[0].keys()
    for metric in metrics:
        values = []
        for epoch_values in six.moves.zip_longest(
                *[h[metric] for h in histories],
                fillvalue=np.nan):
            values.append(np.nanmean(epoch_values))
        averaged[metric] = values
    # Convert {str: [float]} to [{str: float}]
    averaged = [dict(zip(metrics, vals)) for vals in zip(*averaged.values())]
    return averaged
