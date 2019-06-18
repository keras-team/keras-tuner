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

import json
import time
from os import path
from multiprocessing.pool import ThreadPool
from collections import defaultdict
import traceback

import numpy as np

from tensorflow.keras import callbacks

from ..abstractions.tensorflow import TENSORFLOW_UTILS as tf_utils
from .. import utils


class MonitorCallback(callbacks.Callback):

    def __init__(self,
                 tuner,
                 trial,
                 execution,
                 cloudservice,
                 refresh_interval=2,
                 num_threads=4):
        self.tuner = tuner
        self.trial = trial
        self.execution = execution
        self.cloudservice = cloudservice

        self.last_refresh = -1
        self.refresh_interval = refresh_interval
        self.epoch_history = defaultdict(list)
        self.training_complete = False  # important for the cloudservice
        self.num_threads = num_threads
        self.start_time = int(time.time())

    def on_batch_end(self, batch, logs={}):
        for name, value in logs.items():
            value = float(value)
            self.epoch_history[name].append(value)
            self.execution.per_batch_metrics.update(name, value)
        self._report_status()

    def on_epoch_end(self, epoch, logs={}):
        # update epoch counters
        self.execution.epochs_seen += 1

        objective = utils.canonicalize_metric_name(self.tuner.objective)

        # update metrics and checkpoint if needed
        for name, value in logs.items():
            improved = self.execution.per_epoch_metrics.update(name, value)
            name = utils.canonicalize_metric_name(name)
            if objective == name and improved:
                self._checkpoint_model()
                self._write_result_file()

        # reset epoch history
        self.epoch_history = defaultdict(list)

        # update status
        self._report_status(force=True)

    def on_train_end(self, logs={}):
        # Update tracker of averaged metrics on Trial
        if len(self.trial.executions) == 1:
            for name in self.execution.per_epoch_metrics.names:
                direction = self.execution.per_epoch_metrics.directions[name]
                if not self.trial.averaged_metrics.exists(name):
                    self.trial.averaged_metrics.register(name, direction)
                self.trial.averaged_metrics.set_history(
                    name, self.execution.per_epoch_metrics.get_history(name))
        else:
            # Need to average.
            for name in self.execution.per_epoch_metrics.names:
                direction = self.execution.per_epoch_metrics.directions[name]
                histories = []
                for execution in self.trial.executions:
                    histories.append(
                        execution.per_epoch_metrics.get_history(name))
                if len(set(len(h) for h in histories)) != 1:
                    raise ValueError(
                        'Inconsistent metric history length '
                        'across executions for %s' % (name,))
                self.trial.averaged_metrics.set_history(
                    name, list(np.average(histories, axis=0)))

        if len(self.trial.executions) == self.tuner.executions_per_trial:
            # Update tracker of best metrics on Tuner
            for name in self.trial.averaged_metrics.names:
                if not self.tuner.best_metrics.exists(name):
                    direction = self.trial.averaged_metrics.directions[name]
                    self.tuner.best_metrics.register(name, direction)
                self.tuner.best_metrics.update(
                    name, self.trial.averaged_metrics.get_best_value(name))

        self.training_complete = True
        self._report_status(force=True)
        self._write_result_file()

    def _checkpoint_model(self):
        """Checkpoint model"""
        prefix = self._get_filename_prefix()
        base_filename = prefix

        tmp_path = path.join(self.tuner._host.tmp_dir,
                             path.basename(prefix))

        try:
            tf_utils.save_model(self.model,
                                base_filename,
                                tmp_path=tmp_path,
                                export_type='keras')
        except:
            traceback.print_exc()
            write_log('Failed.')
            exit(0)

    def _make_status(self):
        status = {
            'update_time': int(time.time()),
            'tuner': self.tuner.get_status(),
            'trial': self.trial.get_status(),
            'execution': self.execution.get_status(),
        }
        return status

    def _write_result_file(self):
        """Record results - one file per trial"""
        status = self._make_status()

        status_json = json.dumps(status)
        prefix = self._get_filename_prefix(with_execution_info=False)
        # don't do a os.join as it is just appending a suffix
        fname = prefix + '-results.json'
        tf_utils.write_file(fname, status_json)

        # send result to the cloud service
        if self.cloudservice.enabled:
            self.cloudservice.send_results(status)

    def _report_status(self, force=False):
        """Update the status.json file."""
        delta = time.time() - self.last_refresh
        if delta < self.refresh_interval and not force:
            return
        # FIXME: can we make it async?
        # self.thread_pool.apply_async(self._report_status_worker)
        # getting stats
        status = self._make_status()

        # needed for cloudservice
        status['training_complete'] = self.training_complete
        status['epoch_history'] = self.epoch_history
        status_json = json.dumps(status)

        # write on disk
        fname = path.join(self.tuner._host.results_dir, 'status.json')
        tf_utils.write_file(fname, status_json)

        # send status to cloudservice
        if self.cloudservice and self.cloudservice.enabled:
            self.cloudservice.send_status(status)

        self.last_refresh = time.time()

    def _get_filename_prefix(self, with_execution_info=True):
        """Build dir/filename prefix based on the trial & execution."""
        prefix = '%s-%s' % (self.tuner.project_name,
                            self.trial.trial_id)
        if with_execution_info:
            prefix += '-%s' % self.execution.execution_id
        return path.join(self.tuner._host.results_dir, prefix)
