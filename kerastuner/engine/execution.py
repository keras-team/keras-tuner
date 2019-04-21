from __future__ import absolute_import, division, print_function

from copy import deepcopy
import time
from os import path

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Optimizer
from tensorflow.python.lib.io import file_io  # allows to write to GCP or local

from .tunercallback import TunerCallback
from kerastuner.states import ExecutionState

class Execution(object):
    """Model Execution class. Each Model instance can be executed N time"""

    def __init__(self, model, instance_state, tuner_state, cloudservice):

        self.state = ExecutionState()
        self.instance_state = instance_state
        self.tuner_state = tuner_state
        self.cloudservice = cloudservice

        # Model recreaction
        config = model.get_config()
        if isinstance(model, Sequential):
            self.model = Sequential.from_config(config)
        else:
            self.model = keras.Model.from_config(config)

        optimizer_config = tf.keras.optimizers.serialize(model.optimizer)
        optimizer = tf.keras.optimizers.deserialize(optimizer_config)
        self.model.compile(optimizer=optimizer, metrics=model.metrics,
                           loss=model.loss, loss_weights=model.loss_weights)

    def fit(self, x, y, **kwargs):
        """Fit a given model
        Note: This wrapper around Keras fit allows to handle multi-gpu support and use fit or fit_generator
        """
        tcb = TunerCallback(self.instance_info, self.key_metrics,
                            self.meta_data, self.checkpoint, self.backend)
        callbacks = kwargs.get('callbacks')
        if callbacks or self.callback_fn:
            callbacks = copy.deepcopy(callbacks)
            for callback in callbacks:
                # patching tensorboard log dir
                if 'TensorBoard' in str(type(callback)):
                    tensorboard_idx = "%s-%s-%s-%s" % (
                        self.meta_data['project'], self.meta_data['architecture'], self.meta_data['instance'], self.meta_data['execution'])
                    callback.log_dir = path.join(
                        callback.log_dir, tensorboard_idx)

                if self.callback_fn:
                    callbacks.extend(self.callback_fn(self.execution_info))

            callbacks.append(tcb)
        else:
            callbacks = [tcb]

        kwargs['callbacks'] = callbacks

        if self.keras_function == 'fit':
            results = self.model.fit(x, y, **kwargs)
        elif self.keras_function == 'fit_generator':
            results = self.model.fit_generator(x, **kwargs)
        else:
            raise ValueError(
                "Unknown keras function requested ", self.keras_function)

        return results

    def record_results(self, results):
        "Record execution results"

        # History
        history = {}  # need to cast to float for serialization purpose
        for metric, values in results.history.items():
            history[metric] = []
            for value in values:
                history[metric].append(float(value))
        self.history = history

        self.num_epochs = len(self.history['loss'])
        self.ts = int(time.time())

        # generic metric recording
        self.metrics = {}
        for metric, data in self.history.items():
            metric_results = {
                'min': min(data),
                'max': max(data)
            }
            self.metrics[metric] = metric_results
