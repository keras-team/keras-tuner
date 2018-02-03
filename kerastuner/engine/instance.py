import time
import json
import numpy as np
from os import path
from collections import defaultdict
from tensorflow.python.lib.io import file_io # allows to write to GCP or local

from execution import InstanceExecution


class Instance(object):
  """Model instance class."""

  def __init__(self, model, idx, num_gpu, gpu_mem):
    self.ts = int(time.time())
    self.training_size = -1
    self.model = model
    self.idx = idx
    self.num_gpu = num_gpu
    self.gpu_mem = gpu_mem
    self.ts = int(time.time())
    self.executions = []


  def fit(self, x, y, **kwargs):
    "Fit an execution of the model"
    self.training_size = len(y)
    execution = self.__new_execution()
    results = execution.fit(x, y, **kwargs)
    # compute execution level metrics
    execution.record_results(results)
    return results

  def __new_execution(self):
    execution = InstanceExecution(self.model, self.num_gpu, self.gpu_mem)
    self.executions.append(execution)
    return execution

  def record_results(self, output_dir, save_models=True):
    """Record training results
    Args
      output_dir (str): either a local or cloud directory where to store results
      save_models (bool): save the trained models?
    """

    results = {
        "idx": self.idx,
        "ts": self.ts,
        "training_size": self.training_size,
        "num_executions": len(self.executions),
        "model": self.model.to_json()
    }

    # collecting executions data
    exec_metrics = defaultdict(lambda : defaultdict(list))
    executions = [] # execution data
    for execution in self.executions:
      for metric, data in execution.metrics.items():
        exec_metrics[metric]['min'].append(execution.metrics[metric]['min'])
        exec_metrics[metric]['max'].append(execution.metrics[metric]['max'])

      execution = {
        "num_epochs": execution.num_epochs,
        "history": execution.history,
        "loss_fn": execution.model.loss,
        "loss_weigths": execution.model.loss_weights,
        #FIXME record optimizer parameters
        #"optimizer": execution.model.optimizer
      }
      executions.append(execution)
    results['executions'] = executions

    # aggregating statistics
    metrics = defaultdict(dict)
    for metric in exec_metrics.keys():
      for direction, data in exec_metrics[metric].items():
        metrics[metric][direction] = {
          "min": np.min(data),
          "max": np.max(data),
          "mean": np.mean(data),
          "median": np.median(data)
        }
    results['metrics'] = metrics
    
    # Usual metrics reported as top fields for their median values
    top_metrics = [('loss', 'min'), ('val_loss', 'min'), ('acc', 'max'), ('val_acc', 'max')]
    for tm in top_metrics:
      if tm[0] in metrics:
        results[tm[0]] = metrics[tm[0]][tm[1]]['median']

    #FIXME save model

    fname = '%s-%s-results.json' % (self.idx, self.training_size)
    out_path = path.join(output_dir, fname)
    with file_io.FileIO(out_path, 'w') as outfile:
      outfile.write(json.dumps(results))
    return results