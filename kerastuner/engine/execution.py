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

from copy import deepcopy
import time
from os import path
import math

import tensorflow as tf

from tensorflow.keras.models import model_from_config  # nopep8 pylint: disable=import-error
from tensorflow.keras.optimizers import Optimizer  # nopep8 pylint: disable=import-error
from tensorflow.keras.utils import deserialize_keras_object
from tensorflow.keras.utils import serialize_keras_object

from ..callbacks import MonitorCallback
from ..callbacks import DisplayCallback
from ..abstractions.display import fatal
from . import metrics_tracking


class Execution(object):
    """Model Execution class.

    Each Model can be executed N time.

    Not to be subclassed.
    """

    def __init__(self, execution_id, model,
                 trial=None, tuner=None, cloudservice=None):
        self.execution_id = execution_id
        self.model = model
        self.trial = trial
        self.tuner = tuner
        self.cloudservice = cloudservice
        self.history = None

        self._start_time = int(time.time())
        # Number of epochs seen so far
        self.epochs_seen = 0
        # Number of epochs this execution will do in total
        self.max_epochs = 1
        self.batch_size = None
        self.max_steps = None

        self.per_batch_metrics = metrics_tracking.MetricsTracker(model.metrics)
        self.per_epoch_metrics = metrics_tracking.MetricsTracker(model.metrics)

    def run(self, *fit_args, **fit_kwargs):
        self.max_epochs = fit_kwargs.get('epochs', 1)
        self.batch_size = fit_kwargs.get('batch_size', None)
        self.max_steps = self._get_max_steps(*fit_args, **fit_kwargs)

        reinitialize_model(self.model)

        # Tuner callbacks
        monitor_callback = MonitorCallback(self.tuner,
                                           self.trial,
                                           self,
                                           self.cloudservice)
        display_callback = DisplayCallback(self.tuner,
                                           self.trial,
                                           self,
                                           self.cloudservice)

        # Deepcopy and patch callbacks if needed
        callbacks = fit_kwargs.get('callbacks')
        if callbacks:
            callbacks = deepcopy(callbacks)
            for callback in callbacks:
                # patching tensorboard log dir
                if callback.__class__.__name__ == 'TensorBoard':
                    tensorboard_idx = monitor_callback._get_filename_prefix()
                    callback.log_dir = path.join(callback.log_dir,
                                                 tensorboard_idx)
        else:
            callbacks = []

        # Add tuner callbacks -- display_callback must be last
        callbacks.append(monitor_callback)
        callbacks.append(display_callback)

        # Replacing callback with the new list
        fit_kwargs['callbacks'] = callbacks

        # Override verbosity
        fit_kwargs['verbose'] = 0

        history = self.model.fit(*fit_args, **fit_kwargs)
        self.history = history
        return history

    def get_status(self):
        # TODO
        pass

    @property
    def eta(self):
        elapsed_time = int(time()) - self._start_time
        time_per_epoch = elapsed_time / max(self.epochs_seen, 1)
        return int(self.max_epochs * time_per_epoch)

    def _get_max_steps(self, *fit_args, **fit_kwargs):
        if fit_args:
            x = tf.nest.flatten(fit_args)[0]
        else:
            x = tf.nest.flatten(fit_kwargs['x'])[0]
        batch_size = fit_kwargs.get('batch_size', 32)
        if hasattr(x, '__len__'):
            return math.ceil(float(len(x)) / batch_size)
        return fit_kwargs.get('steps')


def reinitialize_model(model):
    # TODO
    pass
