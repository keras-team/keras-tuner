import time
import numpy as np
from collections import defaultdict
from keras.models import clone_model


class InstanceExecution(object):
  """Model Execution class. Each Model instance can be executed N time"""

  def __init__(self, model):
    #final value
    self.acc = -1
    self.loss = -1
    self.val_acc = -1
    self.val_loss = -1
    self.num_epochs = -1
    # keep a separated model per instance
    self.model = clone_model(model)
    # This is directly using Keras model class attribute - I wish there is a better way 
    self.model.compile(optimizer=model.optimizer, loss=model.loss, metrics=model.metrics, loss_weights=model.loss_weights)

  def record_results(self, results):
    "Record execution results"
    self.history = results.history
    self.loss = min(self.history['loss'])
    self.num_epochs = len(self.history)
    
    if 'acc' in self.history:
      self.acc = max(self.history['acc'])
    if 'val_acc' in self.history:
      self.val_acc = max(self.history['val_acc'])
    if 'val_loss' in self.history:
      self.val_loss = min(self.history['val_loss'])

class Instance(object):
  """Model instance class."""

  def __init__(self, model, idx):
    self.model = model
    self.idx = idx
    self.ts = int(time.time())
    self.executions = []

  def new_execution(self):
    execution = InstanceExecution(self.model)
    self.executions.append(execution)
    return execution

  def get_stats(self):
    "Return statistics about executions"

    # collecting data
    exec_results = defaultdict(list)
    for execution in self.executions:
      exec_results['acc'].append(execution.acc)
      exec_results['loss'].append(execution.loss)
      if execution.val_acc != -1:
        exec_results['val_acc'].append(execution.acc)
        exec_results['val_loss'].append(execution.loss)
      
    # aggregation
    stats = {}
    for metric, data in exec_results.items():
      stats[metric] = {
        "idx": self.idx,
        "model": self.model.to_json(),
        "num_executions": len(data),
        "min": np.min(data),
        "max": np.max(data),
        "mean": np.mean(data),
        "median": np.median(data)
      }
    
    return stats

  def to_json():
    print "FIXME"
    return