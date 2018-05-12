import time
import copy
from termcolor import cprint
from keras.models import clone_model
from keras.utils import multi_gpu_model
from os import path
class InstanceExecution(object):
  """Model Execution class. Each Model instance can be executed N time"""

  def __init__(self, model, idx, model_name, num_gpu, gpu_mem, display_model, display_info):
    self.ts = int(time.time())
    self.idx = idx
    self.model_name = model_name
    self.num_epochs = -1
    self.num_gpu = num_gpu
    self.gpu_mem = gpu_mem
    self.display_model = display_model
    self.display_info = display_info
    # keep a separated model per instance
    self.model = clone_model(model)
    # This is directly using Keras model class attribute - I wish there is a better way 
    self.model.compile(optimizer=model.optimizer, loss=model.loss, metrics=model.metrics, loss_weights=model.loss_weights)

  def fit(self, x, y, **kwargs):
      """Fit a given model 
      Note: This wrapper around Keras fit allows to handle multi-gpu support
      """
      
      if self.display_model == 'base' or self.display_model == 'both':
        self.model.summary()

      if self.num_gpu > 1:
        model = multi_gpu_model(self.model, gpus=self.num_gpu)
        model.compile(optimizer=self.model.optimizer, loss=self.model.loss, metrics=self.model.metrics, loss_weights=self.model.loss_weights)
        if self.display_model == 'multi-gpu' or self.display_model == 'both':
          self.model.summary()
      else:
        model = self.model

      # optimize batch_size for gpu memory if needed
      if self.gpu_mem >= 1:
        mem = self.gpu_mem - 1# trying to do more result sometime in OOM
      else:
        # optimize for available system memory
        import psutil
        memory = psutil.virtual_memory()
        mem = int(memory.available / 1073741824) - 1
      batch_size, num_params = self.__compute_batch_size(self.model, mem, len(x))
      if self.display_info:
        cprint("|-batch size is:%d" % batch_size, 'blue')
        cprint("|-model size is:%d" % num_params, 'cyan')
      callbacks = kwargs.get('callbacks')
      if callbacks:
            callbacks = copy.deepcopy(callbacks)
            for callback in callbacks:
              # patching tensorboard log dir
              if 'TensorBoard' in str(type(callback)):
                tensorboard_idx = "%s-%s-%s" % (self.model_name, self.idx, self.ts)
                callback.log_dir = path.join(callback.log_dir, tensorboard_idx)
            kwargs['callbacks'] = callbacks
      results = model.fit(x, y, batch_size=batch_size, **kwargs) 
      return results

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

  def __compute_batch_size(self, model, memory, max_size):
    "Find the largest batch size usuable so we maximize ressources usage"
    batch_size =  16
    while 1:
      bs = batch_size + 2
      if bs >= max_size:
        break
      memory_needed, model_num_params = self.__get_model_memory_usage(model, bs)
      if  memory_needed > memory:
        break
      batch_size = bs
    return batch_size, model_num_params

  def __get_model_memory_usage(self, model, batch_size):
    "comput the memory usage for a given model and batch "
    import numpy as np
    from keras import backend as K

    shapes_mem_count = 0
    for l in model.layers:
        single_layer_mem = 1
        for s in l.output_shape:
            if s is None:
                continue
            single_layer_mem *= s
        shapes_mem_count += single_layer_mem

    trainable_count = np.sum([K.count_params(p) for p in set(model.trainable_weights)])
    non_trainable_count = np.sum([K.count_params(p) for p in set(model.non_trainable_weights)])

    total_memory = 4.0*batch_size*(shapes_mem_count + trainable_count + non_trainable_count)
    gbytes = np.round(total_memory / (1024.0 ** 3), 3)
    #print("train count ", trainable_count, "mem per instance", total_memory, "gbytes ", gbytes)
    return gbytes, trainable_count