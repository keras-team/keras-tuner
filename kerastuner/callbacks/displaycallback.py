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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from time import time
from collections import defaultdict
from multiprocessing.pool import ThreadPool

import tensorflow.keras as keras  # pylint: disable=import-error
from tensorflow.keras import callbacks

from ..abstractions.display import write_log, fatal, info, section, highlight
from ..abstractions.display import subsection, progress_bar
from ..abstractions.display import colorize_row, display_table


class DisplayCallback(callbacks.Callback):

    def __init__(self, tuner, trial, execution, cloudservice):
        self.tuner = tuner
        self.trial = trial
        self.execution = execution
        self.cloudservice = cloudservice

        self.executions_seen = len(trial.executions)
        self.max_executions = tuner.executions_per_trial

        # model tracking
        self.current_epoch = 0
        self.max_epochs = execution.max_epochs

        # epoch tracking
        self.cpu_usage = []
        self.gpu_usage = []
        self.batch_history = defaultdict(list)
        self.epoch_pbar = None
        self.max_steps = execution.max_steps

    def on_train_begin(self, logs=None):
        # new model summary
        if self.executions_seen == 1:
            section('New model')
            self.trial.summary()

        # execution info if needed
        if self.max_executions > 1:
            subsection('Execution %d/%d' % (self.executions_seen,
                                            self.max_executions))

    def on_train_end(self, logs=None):
        # train summary
        if self.executions_seen == self.max_executions:
            curr = self.trial.averaged_metrics
            best = self.tuner.best_metrics
            rows = [['Name', 'Best model', 'Current model']]
            for name in best.names:
                best_value = round(best.get_best_value(name), 4)
                curr_value = round(curr.get_best_value(name), 4)
                row = [name, best_value, curr_value]
                if name == self.tuner.objective:
                    if best_value == curr_value:
                        row = colorize_row(row, 'green')
                    else:
                        row = colorize_row(row, 'red')
                rows.append(row)
            display_table(rows)

            # tuning budget exhausted
            if self.tuner.remaining_trials < 1:
                highlight('Hypertuning complete - results in %s' %
                          self.tuner._host.results_dir)
                # FIXME: final summary
            else:
                highlight('%d/%d trial budget left' %
                          (self.tuner.remaining_trials,
                           self.tuner.max_trials))

    def on_epoch_begin(self, epoch, logs=None):
        # reset counters
        self.epoch_history = defaultdict(list)
        self.gpu_usage = []
        self.cpu_usage = []
        self.current_epoch += 1

        # epoch bar
        self.epoch_pbar = progress_bar(total=self.max_steps,
                                       leave=True,
                                       unit='steps')

    def on_epoch_end(self, epoch, logs=None):

        # compute stats
        final_epoch_postfix = {}
        for m, v in logs.items():
            final_epoch_postfix[m] = round(v, 4)

        # epoch bar
        self.epoch_pbar.set_postfix(final_epoch_postfix)
        self.epoch_pbar.close()

    def on_batch_end(self, batch, logs=None):
        logs = logs or {}
        self.epoch_pbar.update(1)

        # computing metric statistics
        for k, v in logs.items():
            self.batch_history[k].append(v)
        avg_metrics = self._avg_metrics(self.batch_history)
        self.epoch_pbar.set_postfix(avg_metrics)

        # create bar desc with updated statistics
        description = ''
        status = self.tuner._host.get_status()
        if len(status['gpu']):
            gpu_usage = [float(gpu['usage']) for gpu in status['gpu']]
            gpu_usage = int(np.average(gpu_usage))
            self.gpu_usage.append(gpu_usage)
            description += '[GPU:%3s%%]' % int(np.average(self.gpu_usage))

        self.cpu_usage.append(int(status['cpu']['usage']))
        description += '[CPU:%3s%%]' % int(np.average(self.cpu_usage))
        description += 'Epoch %s/%s' % (self.current_epoch, self.max_epochs)
        self.epoch_pbar.set_description(description)

    def _avg_metrics(self, metrics):
        agg_metrics = {}
        for metric_name, values in metrics.items():
            if metric_name == 'batch' or metric_name == 'size':
                continue
            agg_metrics[metric_name] = '%.4f' % np.average(values)
        return agg_metrics
