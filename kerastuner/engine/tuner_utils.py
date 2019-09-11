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

import contextlib
import math
from collections import defaultdict
import numpy as np
import time
import random
import hashlib

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


class Display(object):

    def __init__(self, host):
        self.host = host
        self.cpu_usage = []
        self.gpu_usage = []
        self.batch_history = defaultdict(list)
        self.epoch_pbar = None

    def on_trial_begin(self, trial):
        display.section('New model')
        trial.summary()

    def on_trial_end(self,
                     averaged_metrics,
                     best_metrics,
                     objective,
                     remaining_trials,
                     max_trials):
        # train summary
        current = averaged_metrics
        best = best_metrics
        rows = [['Name', 'Best model', 'Current model']]
        for name in best.names:
            best_value = round(best.get_best_value(name), 4)
            current_value = round(current.get_best_value(name), 4)
            row = [name, best_value, current_value]
            if name == objective:
                if best_value == current_value:
                    row = display.colorize_row(row, 'green')
                else:
                    row = display.colorize_row(row, 'red')
            rows.append(row)
        display.display_table(rows)

        # Tuning budget exhausted
        if remaining_trials < 1:
            display.highlight('Hypertuning complete - results in %s' %
                              self.host.results_dir)
            # TODO: final summary
        else:
            display.highlight('%d/%d trials left' %
                              (remaining_trials, max_trials))

    def on_epoch_begin(self, execution, model, epoch, logs=None):
        # reset counters
        self.epoch_history = defaultdict(list)
        self.gpu_usage = []
        self.cpu_usage = []

        # epoch bar
        self.epoch_pbar = display.progress_bar(
            total=execution.max_steps,
            leave=True,
            unit='steps')

    def on_epoch_end(self, execution, model, epoch, logs=None):
        # compute stats
        final_epoch_postfix = {}
        for m, v in logs.items():
            final_epoch_postfix[m] = round(v, 4)

        # epoch bar
        self.epoch_pbar.set_postfix(final_epoch_postfix)
        self.epoch_pbar.close()

    def on_batch_end(self, execution, model, batch, logs=None):
        logs = logs or {}
        self.epoch_pbar.update(1)

        # computing metric statistics
        for k, v in logs.items():
            self.batch_history[k].append(v)
        avg_metrics = self._avg_metrics(self.batch_history)
        self.epoch_pbar.set_postfix(avg_metrics)

        # create bar desc with updated statistics
        description = ''
        host_status = self.host.get_status()
        if len(host_status['gpu']):
            gpu_usage = [float(gpu['usage']) for gpu in host_status['gpu']]
            gpu_usage = int(np.average(gpu_usage))
            self.gpu_usage.append(gpu_usage)
            description += '[GPU:%3s%%]' % int(np.average(self.gpu_usage))

        self.cpu_usage.append(int(host_status['cpu']['usage']))
        description += '[CPU:%3s%%]' % int(np.average(self.cpu_usage))
        description += 'Epoch %s/%s' % (execution.epochs_seen + 1,
                                        execution.max_epochs)
        self.epoch_pbar.set_description(description)

    def _avg_metrics(self, metrics):
        agg_metrics = {}
        for metric_name, values in metrics.items():
            if metric_name == 'batch' or metric_name == 'size':
                continue
            agg_metrics[metric_name] = '%.4f' % np.average(values)
        return agg_metrics


def format_execution_id(i, executions_per_trial):
    execution_id_length = math.ceil(
        math.log(executions_per_trial, 10))
    execution_id_template = '%0' + str(execution_id_length) + 'd'
    execution_id = execution_id_template % i
    return execution_id


@contextlib.contextmanager
def maybe_distribute(distribution_strategy):
    if distribution_strategy is None:
        yield
    else:
        with distribution_strategy.scope():
            yield
