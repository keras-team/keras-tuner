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
