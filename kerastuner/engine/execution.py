from __future__ import absolute_import, division, print_function

from copy import deepcopy
import time
from os import path
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model  # nopep8 pylint: disable=import-error
from tensorflow.keras.optimizers import Optimizer  # nopep8 pylint: disable=import-error

from kerastuner.callbacks import MonitorCallback
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

        # Model recreaction
        config = model.get_config()
        if isinstance(model, Sequential):
            self.model = Sequential.from_config(config)
        else:
            self.model = Model.from_config(config)

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
        # creating the callback need to track training progress
        monitorcallback = MonitorCallback(self.tuner_state,
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

            callbacks.append(monitorcallback)
        else:
            callbacks = [monitorcallback]

        kwargs['callbacks'] = callbacks

        if self.tuner_state.keras_function == 'fit':
            results = self.model.fit(x, y, **kwargs)
        elif self.tuner_state.keras_function == 'fit_generator':
            results = self.model.fit_generator(x, **kwargs)
        else:
            fatal("Unknown keras function requested ",
                  self.tuner_state.keras_function)
        return results
