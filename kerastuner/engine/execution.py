import time
from termcolor import cprint
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

      # optimize batch_size for gpu memory if needed
      if self.gpu_mem > 1:
        mem = self.gpu_mem 
      else:
        # optimize for available system memory
        import psutil
        memory = psutil.virtual_memory()
        mem = int(memory.available / 1073741824) - 1
      print("mem:%s, max_size:%s" % (mem, len(x)))
      batch_size = self.__compute_batch_size(self.model, mem, len(x))
      cprint("|-batch_size is:%d" % batch_size, 'cyan')

      return model.fit(x, y, batch_size=batch_size, **kwargs) 

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
    return batch_size

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