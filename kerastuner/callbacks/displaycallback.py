from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from time import time
from collections import defaultdict
from multiprocessing.pool import ThreadPool
from kerastuner.abstractions.display import write_log, fatal
import tensorflow.keras as keras  # pylint: disable=import-error


from kerastuner import config
from .tunercallback import TunerCallback
from kerastuner.collections import MetricsCollection
from kerastuner.abstractions.display import write_log, info, section, highlight
from kerastuner.abstractions.display import subsection, progress_bar
from kerastuner.abstractions.display import colorize_row, display_table


class DisplayCallback(TunerCallback):

    def __init__(self, tuner_state, instance_state, execution_state,
                 cloudservice):
        super(DisplayCallback, self).__init__(tuner_state, instance_state,
                                              execution_state, cloudservice)
        self.num_executions = len(
            self.instance_state.execution_states_collection)
        self.max_excutions = self.tuner_state.num_executions

        # model tracking
        self.current_epoch = 0
        self.max_epochs = self.execution_state.max_epochs

        # epoch tracking
        self.cpu_usage = []
        self.gpu_usage = []
        self.batch_history = defaultdict(list)
        self.epoch_pbar = None
        self.num_steps = np.floor(self.instance_state.training_size /
                                  self.instance_state.batch_size)

    def on_train_begin(self, logs={}):

        # new model summary
        if not self.num_executions:
            section('New model')
            self.instance_state.summary()
            if self.tuner_state.display_model:
                subsection("Model summary")
                self.model.summary()

        # execution info if needed
        if self.tuner_state.num_executions > 1:
            subsection("Execution %d/%d" % (self.num_executions + 1,
                                            self.max_excutions))

    def on_train_end(self, logs={}):
        # train summary
        if self.num_executions + 1 == self.max_excutions:
            curr = self.instance_state.agg_metrics.to_config()
            best = self.tuner_state.best_instance_config['aggregate_metrics']
            rows = [['Name', 'Best model', 'Current model']]
            for idx, metric in enumerate(best):
                best_value = round(metric['best_value'], 4)
                curr_value = round(curr[idx]['best_value'], 4)
                row = [metric['name'], best_value, curr_value]
                if metric['is_objective']:
                    if best_value == curr_value:
                        row = colorize_row(row, 'green')
                    else:
                        row = colorize_row(row, 'red')
                rows.append(row)
            display_table(rows)

        # tuning budget exhausted
        if self.tuner_state.remaining_budget < 1:
            highlight("Hypertuning complete - results in %s" %
                      self.tuner_state.host.result_dir)
            # FIXME: final summary
        else:
            highlight("%d/%d epochs tuning budget left" %
                      (self.tuner_state.remaining_budget,
                       self.tuner_state.epoch_budget))

    def on_epoch_begin(self, epoch, logs={}):
        # reset counters
        self.epoch_history = defaultdict(list)
        self.gpu_usage = []
        self.cpu_usage = []
        self.current_epoch += 1

        # epoch bar
        self.epoch_pbar = progress_bar(total=self.num_steps, leave=True,
                                       unit='steps')

    def on_epoch_end(self, epoch, logs={}):

        # compute stats
        final_epoch_postfix = {}
        for m, v in logs.items():
            final_epoch_postfix[m] = round(v, 4)

        # epoch bar
        self.epoch_pbar.set_postfix(final_epoch_postfix)
        self.epoch_pbar.close()

    def on_batch_end(self, batch, logs={}):
        self.epoch_pbar.update(1)

        # computing metric statistics
        for k, v in logs.items():
            self.batch_history[k].append(v)
        avg_metrics = self._avg_metrics(self.batch_history)
        self.epoch_pbar.set_postfix(avg_metrics)

        # create bar desc with updated statistics
        description = ""
        status = config._Host.get_status()
        if len(status['gpu']):
            gpu_usage = [float(gpu["usage"]) for gpu in status['gpu']]
            gpu_usage = int(np.average(gpu_usage))
            self.gpu_usage.append(gpu_usage)
            description += "[GPU:%3s%%]" % int(np.average(self.gpu_usage))

        self.cpu_usage.append(int(status["cpu"]["usage"]))
        description += "[CPU:%3s%%]" % int(np.average(self.cpu_usage))
        description += "Epoch %s/%s" % (self.current_epoch, self.max_epochs)
        self.epoch_pbar.set_description(description)

    def _avg_metrics(self, metrics):
        "Aggregate metrics"
        agg_metrics = {}
        for metric_name, values in metrics.items():
            if metric_name == "batch" or metric_name == "size":
                continue
            agg_metrics[metric_name] = "%.4f" % np.average(values)
        return agg_metrics
