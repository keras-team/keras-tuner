import keras
import time
import sys
from termcolor import cprint
from collections import defaultdict
from os import path
import json
import numpy as np
from tensorflow.python.lib.io import file_io # allows to write to GCP or local

from . import backend


class TunerCallback(keras.callbacks.Callback):
    "Monitoring callback"
    def __init__(self, info, key_metrics, meta_data, checkpoint, log_interval=30):
        """
        Args:
        log_interval: interval of time in second between the execution stats are written on disk
        """
        self.info = info
        self.key_metrics = []
        for k in key_metrics:
            self.key_metrics.append(k[0])
        
        self.meta_data = meta_data
        self.start_ts = int(time.time())
        self.last_write = time.time()
        self.current_epoch_history = defaultdict(list)
        self.current_epoch_key_metrics = defaultdict(list)
        self.history = defaultdict(list)
        self.history_key_metrics = defaultdict(list)
        self.log_interval = log_interval
        self.training_complete = False

        self.checkpoint = checkpoint
        if checkpoint['enable']:
            if checkpoint['mode'] == "max":
                self.checkpoint_current_value = -sys.maxsize -1
            else:
                self.checkpoint_current_value = sys.maxsize

    def on_train_begin(self, logs={}):
        return

    def on_train_end(self, logs={}):
        self.training_complete = True
        self._log()
        return

    def on_epoch_begin(self, epoch, logs={}):
        #clearing epoch
        self.current_epoch_history = defaultdict(list)
        self.current_epoch_key_metrics = defaultdict(list)
        return

    def on_epoch_end(self, epoch, logs={}):

        logs = self._compute_avg_accuracy(logs) # for multi-points
        for k,v in logs.items():
            self.history[k].append(v)
            if k in self.key_metrics:
                self.history_key_metrics[k].append(v)


            # checkpointing model if needed
            update  = False
            if k == self.checkpoint['metric'] and self.checkpoint['enable']:
                if self.checkpoint['mode'] == "min" and v < self.checkpoint_current_value:
                    #cprint('diff:%s'% round(self.checkpoint_current_value - v, 4), 'yellow' )
                    word = "decreased"
                    update = True
                elif self.checkpoint['mode'] == "max" and v > self.checkpoint_current_value:
                    word = "increased"
                    update = True
            
            if update:
                cprint("[INFO] Saving model %s %s from %s to %s" % (k, word, round(self.checkpoint_current_value, 4), round(v, 4)), 'yellow')
                self.checkpoint_current_value = v
                self._save_model()

        self._log()
        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        for k,v in logs.items():
            self.current_epoch_history[k].append(v)
            if k in self.key_metrics:
                self.current_epoch_key_metrics[k].append(v)
            self._log()
        return

    def _save_model(self):
        """Save model
            
            note: we save model and weights separately because the model might be trained with multi-gpu which use a different architecture
        """
        local_dir = self.meta_data['server']['local_dir']

        #config
        prefix = '%s-%s-%s-%s' % (self.meta_data['project'], self.meta_data['architecture'], self.meta_data['instance'], self.meta_data['execution'])
        config_fname = "%s-config.json" % (prefix)
        local_path = path.join(local_dir, config_fname)
        with file_io.FileIO(local_path, 'w') as output:
            output.write(self.model.to_json())
        backend.cloud_save(local_path=local_path, ftype='config', meta_data=self.meta_data)

        # weights
        weights_fname = "%s-weights.h5" % (prefix)
        local_path = path.join(local_dir, weights_fname)
        self.model.save_weights(local_path)
        backend.cloud_save(local_path=local_path, ftype='weights', meta_data=self.meta_data)
        return

    def _compute_avg_accuracy(self, logs):
        """Compute average accuracy metrics for multi-points if needed
        Args:
            logs: epoch_end logs
        returns
            logs: epoch_end logs with additional metrics
        """
        #Adding combined accuracy metrics for multi-output if needed
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
                logs['val_avg_accuracy'] = round(total_val_acc / float(num_val_acc_metrics), 4)
        return logs


    def _log(self):
        # If not enough time has passed since the last upload, skip it. However,
        # don't skip it if it's the last write we make about this training.
        if (time.time() - self.last_write) < self.log_interval:
            if not self.training_complete:
                return

        ts = time.time()
        results  = self.info

        #ETA
        elapsed_time = ts - self.start_ts
        num_epochs = len(self.history)
        
        
        results['num_epochs'] = num_epochs
        results['time_per_epoch'] = int(elapsed_time / float(max(num_epochs, 1)))
        results['eta'] = (self.meta_data['tuner']['max_epochs'] - num_epochs) * results['time_per_epoch']


        #epoch data must be aggregated
        current_epoch_metrics = {}
        current_epoch_key_metrics = {}
        for k, values in self.current_epoch_history.items():
            if k in ['batch', 'size']:
                continue
            avg = float(np.average(values))
            current_epoch_metrics[k] = avg
            if k in self.current_epoch_key_metrics:
                current_epoch_key_metrics[k] = avg
        if 'size' in self.current_epoch_history and len(self.current_epoch_history['size']):
            results['batch_size'] = self.current_epoch_history['size'][0]
        results['current_epoch_metrics'] = current_epoch_metrics
        results['current_epoch_key_metrics'] = current_epoch_key_metrics

        results['history'] = self.history
        results['ts'] = {"start": self.start_ts, "stop": int(time.time())}
        results['num_epochs'] = num_epochs
        results['training_complete'] = self.training_complete
        results['meta_data'] = self.meta_data

        fname = '%s-%s-%s-%s-execution.json' % (self.meta_data['project'], self.meta_data['architecture'], self.meta_data['instance'], self.meta_data['execution'])
        local_path = path.join(self.meta_data['server']['local_dir'], fname)
        with file_io.FileIO(local_path, 'w') as outfile:
            outfile.write(json.dumps(results))

        backend.cloud_save(local_path=local_path, ftype='execution', meta_data=self.meta_data )
        self.last_write = time.time()
