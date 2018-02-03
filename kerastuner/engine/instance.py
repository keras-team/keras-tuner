import time
import numpy as np
from collections import defaultdict
from execution import InstanceExecution

class Instance(object):
  """Model instance class."""

  def __init__(self, model, idx, num_gpu, gpu_mem):
    self.model = model
    self.idx = idx
    self.num_gpu = num_gpu
    self.gpu_mem = gpu_mem
    self.ts = int(time.time())
    self.executions = []

  def new_execution(self):
    execution = InstanceExecution(self.model, self.num_gpu, self.gpu_mem)
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

  def to_json(self):
    print "FIXME"
    return