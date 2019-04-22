from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from time import time
from multiprocessing.pool import ThreadPool
from kerastuner.abstractions.display import write_log, fatal
import tensorflow.keras as keras  # pylint: disable=import-error


class TunerCallback(keras.callbacks.Callback):

    def __init__(self, tuner_state, instance_state, execution_state,
                 cloudservice):
        self.tuner_state = tuner_state
        self.instance_state = instance_state
        self.execution_state = execution_state
        self.cloudservice = cloudservice
        self.start_time = int(time())
        self.training_complete = False

    def on_train_begin(self, logs={}):
        pass

    def on_train_end(self, logs={}):
        pass

    def on_epoch_begin(self, epoch, logs={}):
        pass

    def on_epoch_end(self, epoch, logs={}):
        pass

    def on_batch_begin(self, batch, logs={}):
        pass

    def on_batch_end(self, batch, logs={}):
        pass
