import time
import copy
import numpy as np
from termcolor import cprint
import keras
from os import path
from tensorflow.python.lib.io import file_io # allows to write to GCP or local
from . import backend
from .tunercallback import TunerCallback

class InstanceExecution(object):
  """Model Execution class. Each Model instance can be executed N time"""

  def __init__(self, model, idx, meta_data, num_gpu, display_model, display_info, instance_info, key_metrics, keras_function, checkpoint, callback_fn):
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
    self.model = keras.models.clone_model(model)
    # This is directly using Keras model class attribute - I wish there is a better way 
    self.model.compile(optimizer=model.optimizer, loss=model.loss, metrics=model.metrics, loss_weights=model.loss_weights)
    self.instance_info = instance_info
    self.key_metrics = key_metrics
    self.keras_function = keras_function
    self.callback_fn = callback_fn
      
    # reflected to the callback_fn which is a user function and therefore must be documented / decoupled 
    self.execution_info = {}

    for k in ['project', 'architecture', 'instance', 'execution']:
        self.execution_info[k] = self.meta_data[k]

    for k in ['training_size', 'validation_size', 'batch_size', 'model_size', 'hyper_parameters']:
        self.execution_info[k] = self.instance_info[k]



    if (self.display_model == 'base' or self.display_model == 'both') and self.display_info :
      self.model.summary()

    if self.num_gpu > 1:
      model = keras.utils.multi_gpu_model(self.model, gpus=self.num_gpu)
      model.compile(optimizer=self.model.optimizer, loss=self.model.loss, metrics=self.model.metrics, loss_weights=self.model.loss_weights)
      if (self.display_model == 'multi-gpu' or self.display_model == 'both') and self.display_info:
        self.model.summary()
    else:
      model = self.model

  def fit(self, x, y, **kwargs):
      """Fit a given model 
      Note: This wrapper around Keras fit allows to handle multi-gpu support and use fit or fit_generator
      """
      tcb = TunerCallback(self.instance_info, self.key_metrics, self.meta_data, self.checkpoint)
      callbacks = kwargs.get('callbacks')
      if callbacks or self.callback_fn:
            if callbacks:
                callbacks = copy.deepcopy(callbacks)
            else:
                callbacks = []

            for callback in callbacks:
              # patching tensorboard log dir
              if 'TensorBoard' in str(type(callback)):
                tensorboard_idx = "%s-%s-%s-%s" % (self.meta_data['project'], self.meta_data['architecture'], self.meta_data['instance'], self.meta_data['execution'])
                callback.log_dir = path.join(callback.log_dir, tensorboard_idx)

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
        raise ValueError("Unknown keras function requested ", self.keras_function)
      return results

  def record_results(self, results):
    "Record execution results"
    
    self.history = results.history
    self.num_epochs = len(self.history)
    self.ts = int(time.time())
    local_dir = self.meta_data['server']['local_dir']

    # generic metric recording 
    self.metrics = {}
    for metric, data in self.history.items():
      metric_results = {
        'min': min(data),
        'max': max(data)
      }
      self.metrics[metric] = metric_results
