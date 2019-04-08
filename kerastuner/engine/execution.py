from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import copy
import numpy as np
from termcolor import cprint
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Optimizer
from os import path
from tensorflow.python.lib.io import file_io  # allows to write to GCP or local
from . import backend
from .tunercallback import TunerCallback


class InstanceExecution(object):
    """Model Execution class. Each Model instance can be executed N time"""

    def __init__(self, model, idx, meta_data, num_gpu, display_model,
                 display_info, instance_info, key_metrics, keras_function,
                 checkpoint, callback_fn, backend):
        self.ts = int(time.time())
        self.idx = idx

        self.meta_data = copy.deepcopy(meta_data)
        self.meta_data['execution'] = self.ts

        self.num_epochs = -1
        self.num_gpu = num_gpu
        self.display_model = display_model
        self.display_info = display_info
        self.checkpoint = checkpoint

        # keep a separated model per instance
        config = model.get_config()
        if isinstance(model, Sequential):
            self.model = Sequential.from_config(config)
        else:
            self.model = keras.Model.from_config(config)
        optimizer_config = model.optimizer.get_config()
        optimizer = model.optimizer.__class__.from_config(optimizer_config)
        self.model.compile(optimizer=optimizer, loss=model.loss,
                           metrics=model.metrics, loss_weights=model.loss_weights)

        self.model = model
        self.instance_info = instance_info
        self.key_metrics = key_metrics
        self.keras_function = keras_function
        self.callback_fn = callback_fn
        self.backend = backend

        # reflected to the callback_fn which is a user function and therefore must be documented / decoupled
        self.execution_info = {}

        for k in ['project', 'architecture', 'instance', 'execution']:
            self.execution_info[k] = self.meta_data[k]

        for k in ['training_size', 'validation_size', 'batch_size', 'model_size', 'hyper_parameters']:
            self.execution_info[k] = self.instance_info[k]

        if (self.display_model == 'base' or self.display_model == 'both') and self.display_info:
            self.model.summary()

        #FIXME compile the model on CPU > recommended to avoid OOO
        if self.num_gpu > 1:
            model = keras.utils.multi_gpu_model(self.model, gpus=self.num_gpu)
            # WARNING: model.compile do NOT return a model
            model.compile(optimizer=optimizer, loss=self.model.loss,
                          metrics=self.model.metrics, loss_weights=self.model.loss_weights)
            self.model = model
            if (self.display_model == 'multi-gpu' or self.display_model == 'both') and self.display_info:
                self.model.summary()

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
