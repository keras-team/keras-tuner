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

from __future__ import absolute_import, division, print_function

from copy import deepcopy
import time
from os import path
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import model_from_config  # nopep8 pylint: disable=import-error
from tensorflow.keras.optimizers import Optimizer  # nopep8 pylint: disable=import-error
from tensorflow.keras.utils import deserialize_keras_object
from tensorflow.keras.utils import serialize_keras_object

from kerastuner.callbacks import MonitorCallback, DisplayCallback
from kerastuner.states import ExecutionState
from kerastuner.abstractions.display import fatal


class Execution(object):
    """Model Execution class. Each Model instance can be executed N time"""

    def __init__(self, model, instance_state, tuner_state, metrics_config,
                 cloudservice):
        self.instance_state = instance_state
        self.tuner_state = tuner_state
        self.cloudservice = cloudservice
        self.state = ExecutionState(tuner_state.max_epochs, metrics_config)

        # Model rereation
        config = serialize_keras_object(model)
        self.model = model_from_config(config)

        optimizer_config = tf.keras.optimizers.serialize(model.optimizer)
        optimizer = tf.keras.optimizers.deserialize(optimizer_config)

        self.model.compile(optimizer=optimizer, metrics=model.metrics,
                           loss=model.loss, loss_weights=model.loss_weights)

    def fit(self, x, y, **kwargs):
        """Fit a given model"""

        # if dry-run just pass
        if self.tuner_state.dry_run:
            self.tuner_state.remaining_budget -= self.tuner_state.max_epochs
            return

        # Tuner callbacks
        # TODO - support validation_split as well as validation data.
        monitorcallback = MonitorCallback(self.tuner_state,
                                          self.instance_state,
                                          self.state, self.cloudservice,
                                          kwargs.get("validation_data", None))

        # display
        displaycallback = DisplayCallback(self.tuner_state,
                                          self.instance_state,
                                          self.state, self.cloudservice)

        # deepcopying and patching callbacks  if needed
        callbacks = kwargs.get('callbacks')
        if callbacks:
            callbacks = deepcopy(callbacks)
            for callback in callbacks:
                # patching tensorboard log dir
                if 'TensorBoard' in str(type(callback)):
                    tensorboard_idx = monitorcallback._get_filename_prefix()
                    callback.log_dir = path.join(callback.log_dir,
                                                 tensorboard_idx)
        else:
            callbacks = []

        # adding tuner callbacks -- displaycallback must be last
        callbacks.append(monitorcallback)
        callbacks.append(displaycallback)

        # replacing callback with the new list
        kwargs['callbacks'] = callbacks
        results = self.model.fit(x, y, **kwargs)
        return results
