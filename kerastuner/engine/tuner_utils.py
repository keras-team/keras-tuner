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

from .. import utils

import math
import numpy as np
import six
import time

import tensorflow as tf
from tensorflow import keras


class TunerStats(object):
    """Track tuner statistics."""

    def __init__(self):
        self.num_generated_models = 0  # overall number of instances generated
        self.num_invalid_models = 0  # how many models didn't work
        self.num_oversized_models = 0  # num models with params> max_params

    def summary(self, extended=False):
        print('Tuning stats')
        for setting, value in self.get_config():
            print(setting + ':', value)

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

    def __init__(self, oracle, verbose=1):
        self.verbose = verbose
        self.oracle = oracle
        self.trial_number = 0

        # Start time for the overall search
        self.search_start = None

        # Start time of the latest trial
        self.trial_start = None

    def on_trial_begin(self, trial):
        if self.verbose >= 1:

            self.trial_number += 1
            print()
            print('Search: Running Trial #{}'.format(self.trial_number))
            print()

            self.trial_start = time.time()
            if self.search_start is None:
                self.search_start = time.time()

            self.show_hyperparameter_table(trial)
            print()

    def on_trial_end(self, trial):
        if self.verbose >= 1:
            utils.try_clear()

            time_taken_str = self.format_time(time.time() - self.trial_start)
            print('Trial {} Complete [{}]'.format(self.trial_number, time_taken_str))

            if trial.score is not None:
                print('{}: {}'.format(self.oracle.objective.name, trial.score))

            print()
            best_trials = self.oracle.get_best_trials()
            if len(best_trials) > 0:
                best_score = best_trials[0].score
            else:
                best_score = None
            print('Best {} So Far: {}'.format(
                self.oracle.objective.name, best_score))

            time_elapsed_str = self.format_time(time.time() - self.search_start)
            print('Total elapsed time: {}'.format(time_elapsed_str))

    def show_hyperparameter_table(self, trial):
        template = "{0:20}|{1:10}|{2:20}"
        best_trials = self.oracle.get_best_trials()
        if len(best_trials) > 0:
            best_trial = best_trials[0]
        else:
            best_trial = None
        if trial.hyperparameters.values:
            print(template.format('Hyperparameter', 'Value', 'Best Value So Far'))
            for hp, value in trial.hyperparameters.values.items():
                if best_trial:
                    best_value = self._format_value(
                        best_trial.hyperparameters.values.get(hp))
                else:
                    best_value = '?'
                print(template.format(hp,
                                      self._format_value(value),
                                      best_value))
        else:
            print('default configuration')

    def format_time(self, t):
        return time.strftime("%Hh %Mm %Ss", time.gmtime(t))

    def _format_value(self, v):
        if isinstance(v, (float)):
            if v >= 1e-4 and v < 10000:
                # e.g.: 0.001032 -> 0.001, 9999 -> 9999
                return '{:.4f}'.format(v).rstrip('0').rstrip('.')
            else:
                # e.g.: 0.00001 -> 1e-5
                a, b = '{:.4e}'.format(v).split('e')
                return 'e'.join([a.rstrip('0').rstrip('.'), b])
        else:
            return str(v)


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
