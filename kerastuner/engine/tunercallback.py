"Callback"
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow import keras
import time
import sys
from collections import defaultdict
from os import path
import json
from tensorflow.python.lib.io import file_io  # allows to write to GCP or local
from copy import copy
from multiprocessing.pool import ThreadPool
from kerastuner.abstractions.display import colorize, print_combined_table, section, highlight
from kerastuner.abstractions.system import System


class TunerCallback(keras.callbacks.Callback):
    "Monitoring callback"

    def __init__(self,
                 info,
                 key_metrics,
                 meta_data,
                 checkpoint,
                 backend,
                 log_interval=2,
                 num_threads=4):
        """
        Args:
        log_interval: interval between the execution stats are written on disk
        """
        self.info = info
        self.key_metrics = {}
        for km in key_metrics:
            self.key_metrics[km[0]] = km[1]
        self.meta_data = meta_data

        self.current_epoch_history = defaultdict(list)
        self.current_epoch_key_metrics = defaultdict(list)
        self.history = defaultdict(list)
        self.history_key_metrics = defaultdict(list)

        self.start_ts = int(time.time())
        self.log_interval = log_interval
        self.last_write = int(time.time())
        self.training_complete = False
        self.system = System()  # computer stats
        self.backend = backend

        self.stats = {}  # track stats per epoch
        self.thread_pool = ThreadPool(num_threads)

        self.checkpoint = checkpoint
        if checkpoint['enable']:
            if checkpoint['mode'] == "max":
                self.cpt_cur_val = -sys.maxsize - 1
            else:
                self.cpt_cur_val = sys.maxsize

    def on_train_begin(self, logs={}):
        return

    def on_train_end(self, logs={}):

        # statistics update
        self.meta_data['statistics']['latest'] = self.stats

        self.training_complete = True
        self._report_status()
        self._display_statistics()
        return

    def on_epoch_begin(self, epoch, logs={}):
        " clearing metric data at epoch end"
        self.current_epoch_history = defaultdict(list)
        self.current_epoch_key_metrics = defaultdict(list)
        self.epoch_start_ts = time.time()
        return

    def on_epoch_end(self, epoch, logs={}):
        logs["epoch_duration"] = time.time() - self.epoch_start_ts

        # for multi-points compute average accuracy
        logs = self._compute_avg_accuracy(logs)

        # metric update
        for k, v in logs.items():

            # must cast v to float for serialization
            self.history[k].append(float(v))
            if k in self.key_metrics:
                self.history_key_metrics[k].append(float(v))

                # update current model performance stats
                if self.key_metrics[k] == 'min':
                    self.stats[k] = min(self.history_key_metrics[k])
                else:
                    self.stats[k] = max(self.history_key_metrics[k])

            # checkpointing model if needed
            update = False
            if k == self.checkpoint['metric'] and self.checkpoint['enable']:
                if self.checkpoint['mode'] == "min" and v < self.cpt_cur_val:
                    word = "decreased"
                    update = True
                elif self.checkpoint['mode'] == "max" and v > self.cpt_cur_val:
                    word = "increased"
                    update = True

            if update:
                highlight("\nSaving improved model %s %s from %s to %s" %
                          (k, word, round(self.cpt_cur_val, 4), round(v, 4)))
                self.cpt_cur_val = v
                self._save_model()

        # update statistics
        self.meta_data['tuner']['remaining_budget'] -= 1
        self.meta_data['statistics']['latest'] = self.stats

        # report status
        self._report_status()
        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        for k, v in logs.items():
            v = float(v)
            self.current_epoch_history[k].append(v)
            if k in self.key_metrics:
                self.current_epoch_key_metrics[k].append(v)

        self._report_status()
        return

    def _save_model(self):
        """Save model

            note: we save model and weights separately because
            the model might be trained with multi-gpu
            which use a different architecture
        """
        local_dir = self.meta_data['server']['local_dir']

        # config
        prefix = '%s-%s-%s-%s' % (
            self.meta_data['project'], self.meta_data['architecture'],
            self.meta_data['instance'], self.meta_data['execution'])
        config_fname = "%s-config.json" % (prefix)
        local_path = path.join(local_dir, config_fname)
        with file_io.FileIO(local_path, 'w') as output:
            output.write(self.model.to_json())
        # FIXME:refactor
        if self.backend:
            self.backend.send_config(self.model.to_json())

        # weights
        weights_fname = "%s-weights.h5" % (prefix)
        local_path = path.join(local_dir, weights_fname)
        self.model.save_weights(local_path)
        # ! don't save weight to the service, instead implment a GS save
        # backend.cloud_save(local_path=local_path,
        # ftype='weights', meta_data=self.meta_data)
        return

    def _compute_avg_accuracy(self, logs):
        """Compute average accuracy metrics for multi-points if needed
        Args:
            logs: epoch_end logs
        returns
            logs: epoch_end logs with additional metrics
        """
        # Adding combined accuracy metrics for multi-output if needed
        num_acc_metrics = 0
        num_val_acc_metrics = 0
        for k in logs.keys():
            if '_accuracy' in k:
                if 'val_' in k:
                    num_val_acc_metrics += 1
                else:
                    num_acc_metrics += 1

        # multi acc metric -> compute average one
        if num_acc_metrics > 1:
            total_acc = 0
            total_val_acc = 0
            for k, v in logs.items():
                if '_accuracy' in k:
                    if 'val_' in k:
                        total_val_acc += v
                    else:
                        total_acc += v
            logs['avg_accuracy'] = round(total_acc / float(num_acc_metrics), 4)
            if num_val_acc_metrics:
                logs['val_avg_accuracy'] = round(
                    total_val_acc / float(num_val_acc_metrics), 4)
        return logs

    def _report_status(self):
        ts = time.time()
        delta = ts - self.last_write
        if delta < self.log_interval and not self.training_complete:
            return

        self.thread_pool.apply_async(self._report_status_worker)

    def _report_status_worker(self):
        "Report tuner status periodically"

        ts = time.time()

        # copy existing meta_data
        status = copy(self.meta_data)
        status['training_complete'] = self.training_complete

        # hypertuning eta
        elapsed_time = int(ts - self.meta_data['tuner']['start_time'])
        epochs = status['tuner']['epoch_budget'] - \
            status['tuner']['remaining_budget']
        time_per_epoch = elapsed_time / max(epochs, 1)
        eta = status['tuner']['remaining_budget'] * time_per_epoch
        status['tuner']['eta'] = eta

        # Current model eta
        elapsed_time = int(ts - self.start_ts)
        epochs = len(self.history['loss'])
        time_per_epoch = elapsed_time / max(epochs, 1)
        eta = (self.meta_data['tuner']['max_epochs'] - epochs) * time_per_epoch

        # model info
        current_model = {
            'elapsed_time': elapsed_time,
            'epochs': epochs,
            'time_per_epoch': time_per_epoch,
            'eta': eta
        }

        status["current_model"] = current_model

        status["batch_metrics"] = self.current_epoch_key_metrics
        status["epoch_metrics"] = self.history_key_metrics
        status["server"].update(self.system.get_status())

        # write on disk
        local_dir = self.meta_data['server']['local_dir']
        fname = path.join(local_dir, 'status.json')
        with file_io.FileIO(fname, 'w') as outfile:
            outfile.write(json.dumps(status))

        # send status to the cloud service
        if self.backend:
            self.backend.send_status(status)

        # update write time
        self.last_write = time.time()

    def _display_statistics(self):
        """ Report statistics at training end
        """
        stats_data = [['Metric', 'Best model', 'Last model']]
        stats = self.meta_data['statistics']
        for metric_name in stats['best'].keys():
            best = round(stats['best'][metric_name], 4)
            last = round(stats['latest'][metric_name], 4)

            # colorize improvement
            if ((self.key_metrics[metric_name] == 'min' and last <= best) or
                    (self.key_metrics[metric_name] == 'max' and last >= best)):
                last = colorize(last, 'white', 'green', 'bright')

            stats_data.append([metric_name, best, last])

        # tuner metrics
        tuner_data = [['Error', 'count']]
        md = self.meta_data['tuner']
        metrics = ['collisions', 'invalid_models', 'over_size_models']
        for k in metrics:
            tuner_data.append([k.replace('_', ' '), md[k]])

        # display both on the same line
        section("Statistics")
        highlight("Trained models: %s" % md['trained_models'])
        print_combined_table([stats_data, tuner_data])
