import time
from keras.models import clone_model
from keras.utils import multi_gpu_model
class InstanceExecution(object):
  """Model Execution class. Each Model instance can be executed N time"""

  def __init__(self, model, num_gpu, gpu_mem):
    self.ts = int(time.time())
    self.num_epochs = -1
    self.num_gpu = num_gpu
    self.gpu_mem = gpu_mem
    # keep a separated model per instance
    self.model = clone_model(model)
    # This is directly using Keras model class attribute - I wish there is a better way 
    self.model.compile(optimizer=model.optimizer, loss=model.loss, metrics=model.metrics, loss_weights=model.loss_weights)

  def fit(self, x, y, **kwargs):
      """Fit a given model 
      Note: This wrapper around Keras fit allows to handle multi-gpu support
      """
      
      if self.num_gpu > 1:
        model = multi_gpu_model(self.model, gpus=self.num_gpu)
        model.compile(loss=self.model.loss_fn, optimizer=self.model.optimizer, metrics=self.model.metrics, 
                      loss_weights=self.model.loss_weights)
      else:
        model = self.model

      return model.fit(x, y, **kwargs) 

  def record_results(self, results):
    "Record execution results"
    
    self.history = results.history
    self.num_epochs = len(self.history)

    # generic metric recording 
    self.metrics = {}
    for metric, data in self.history.items():
      metric_results = {
        'min': min(data),
        'max': max(data)
      }
      self.metrics[metric] = metric_results
