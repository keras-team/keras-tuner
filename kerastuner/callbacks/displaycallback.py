from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from time import time
from multiprocessing.pool import ThreadPool
from kerastuner.abstractions.display import write_log, fatal
import tensorflow.keras as keras  # pylint: disable=import-error


from kerastuner import config
from .tunercallback import TunerCallback
from kerastuner.collections import MetricsCollection
from kerastuner.abstractions.display import write_log, info, section,
from kerastuner.abstractions.display import subsection, progress_bar


class DisplayCallback(TunerCallback):

    def __init__(self, tuner_state, instance_state, execution_state,
                 cloudservice):
        super(DisplayCallback, self).__init__(tuner_state, instance_state,
                                              execution_state, cloudservice)
        self.num_executions = len(self.instance_state.execution_config)
        self.max_excutions = self.tuner_state.num_executions

        # model tracking
        self.max_epochs = self.instance_state.max_epochs
        self.model_pbar = None

        # epoch tracking
        self.epoch_pbar = None
        self.num_steps = np.floor(self.instance_state.training_size /
                                  self.instance_state.batch_size)

    def on_train_begin(self, logs={}):

        # new model summary
        if not self.instance_state.excution_config:
            section('New model')
            self.instance_state.summary()
            if self.tuner_state.display_model:
                subsection("Model summary")
                self.model.summary()

        # execution info if needed
        if self.tuner_state.num_executions > 1:
            subsection("Execution %d/%d" % (self.num_executions,
                                            self.max_excutions))
        # model bar
        self.model_pbar = get_progress_bar(desc="", unit="epochs",
                                           total=self.max_epochs)

    def on_train_end(self, logs={}):
        # model bar
        self.model_pbar.close()

        # tuning budget exhausted
        if self.tuner_state.remaining_budget < 1:
            info("Hypertuning complete - results in %s" %
                 self.tuner_state.host.result_dir)
            # FIXME: final summary

    def on_epoch_begin(self, epoch, logs={}):
        # model bar
        self.model_pbar.update(1)

        # epoch bar
        self.epoch_pbar = get_progress_bar(total=self.num_steps, units='steps',
                                           desc="")

    def on_epoch_end(self, epoch, logs={}):

        # epoch bar
        self.epoch_pbar.close()

    def on_batch_end(self, batch, logs={}):
        # epoch bar
        self.epoch_pbar.update(1)
