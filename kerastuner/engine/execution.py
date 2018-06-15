import time
import copy
import numpy as np
from termcolor import cprint
import keras
from os import path

from .tunercallback import TunerCallback

class InstanceExecution(object):
  """Model Execution class. Each Model instance can be executed N time"""

  def __init__(self, model, idx, model_name, num_gpu, batch_size, display_model, display_info, instance_info, key_metrics, gs_dir):
    self.ts = int(time.time())
    self.idx = idx
    self.model_name = model_name
    self.num_epochs = -1
    self.num_gpu = num_gpu
    self.batch_size = batch_size
    self.display_model = display_model
    self.display_info = display_info
    self.gs_dir = gs_dir
    # keep a separated model per instance
    self.model = keras.models.clone_model(model)
    # This is directly using Keras model class attribute - I wish there is a better way 
    self.model.compile(optimizer=model.optimizer, loss=model.loss, metrics=model.metrics, loss_weights=model.loss_weights)
    self.instance_info = instance_info
    self.key_metrics = key_metrics

  def fit(self, x, y, **kwargs):
      """Fit a given model 
      Note: This wrapper around Keras fit allows to handle multi-gpu support
      """
      
      if (self.display_model == 'base' or self.display_model == 'both') and self.display_info :
        self.model.summary()
      #FIXME: need to be moved to init
      if self.num_gpu > 1:
        model = keras.utils.multi_gpu_model(self.model, gpus=self.num_gpu)
        model.compile(optimizer=self.model.optimizer, loss=self.model.loss, metrics=self.model.metrics, loss_weights=self.model.loss_weights)
        if (self.display_model == 'multi-gpu' or self.display_model == 'both') and self.display_info:
          self.model.summary()
      else:
        model = self.model

      self.instance_info['execution_idx'] = self.ts
      tcb = TunerCallback(self.instance_info, self.key_metrics, gs_dir=self.gs_dir)
      callbacks = kwargs.get('callbacks')
      if callbacks:
            callbacks = copy.deepcopy(callbacks)
            for callback in callbacks:
              # patching tensorboard log dir
              if 'TensorBoard' in str(type(callback)):
                tensorboard_idx = "%s-%s-%s" % (self.model_name, self.idx, self.ts)
                callback.log_dir = path.join(callback.log_dir, tensorboard_idx)
            callbacks.append(tcb)
      else: 
          callbacks = [tcb]
      kwargs['callbacks'] = callbacks
      results = model.fit(x, y, batch_size=self.batch_size, **kwargs) 
      return results

  def record_results(self, results):
    "Record execution results"
    
    self.history = results.history
    self.num_epochs = len(self.history)
    self.ts = int(time.time())

    # generic metric recording 
    self.metrics = {}
    for metric, data in self.history.items():
      metric_results = {
        'min': min(data),
        'max': max(data)
      }
      self.metrics[metric] = metric_results
  
