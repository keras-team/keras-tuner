import time
import json
import numpy as np
from os import path
from collections import defaultdict
from tensorflow.python.lib.io import file_io # allows to write to GCP or local
from termcolor import cprint
from keras import backend as K

from .execution import InstanceExecution


class Instance(object):
  """Model instance class."""

  def __init__(self, model, idx, model_name, num_gpu, batch_size, display_model):
    self.ts = int(time.time())
    self.training_size = -1
    self.model = model
    self.idx = idx
    self.model_name = model_name
    self.num_gpu = num_gpu
    self.batch_size = batch_size
    self.display_model = display_model
    self.ts = int(time.time())
    self.executions = []
    self.model_size = self.__compute_model_size(model)
    self.validation_size = 0
    self.results = None
    
      
  def __compute_model_size(self, model):
    "comput the size of a given model"
    return np.sum([K.count_params(p) for p in set(model.trainable_weights)])

  def fit(self, x, y, resume_execution=False, **kwargs):
    """Fit an execution of the model instance
    Args:
      resume_execution (bool): Instead of creating a new execution, resume training the previous one. Default false.
    """
    self.training_size = len(y)
    if kwargs.get('validation_data'):
      self.validation_size = len(kwargs['validation_data'][1]) 

    if resume_execution and len(self.executions):
      execution = self.executions[-1]
      #FIXME: merge accuracy back
      results = execution.fit(x, y, initial_epoch=execution.num_epochs ,**kwargs)
    else:
      execution = self.__new_execution()
      results  = execution.fit(x, y, **kwargs)
    # compute execution level metrics
    execution.record_results(results)
    return results

  def __new_execution(self):
    num_executions = len(self.executions)
    
    # ensure that info is only displayed once per iteration
    if num_executions > 0:
      display_model = None
      display_info = False
    else:
      display_info = True
      display_model = self.display_model

    execution = InstanceExecution(self.model, self.idx, self.model_name, self.num_gpu, self.batch_size, display_model, display_info)
    self.executions.append(execution)
    return execution

  def __save_to_gs(self, fname, local_dir, gs_dir):
    "Store file remotely in a given GS bucket path"
    local_path = path.join(local_dir, fname)
    remote_path = "%s%s" % (gs_dir, fname)
    cprint("[INFO] Uploading %s to %s" % (local_path, remote_path), 'cyan')
    with file_io.FileIO(local_path, mode='r') as input_f:
      with file_io.FileIO(remote_path, mode='w+') as output_f:
        output_f.write(input_f.read())


  def record_results(self, local_dir, gs_dir=None, save_models=True, prefix='', 
                    key_metrics=[('loss', 'min'), ('acc', 'max')]):
    """Record training results
    Args
      local_dir (str): local saving directory
      gs_dir (str): Google cloud bucket path. Default None
      save_models (bool): save the trained models?
      prefix (str): what string to use to prefix the models
      key_metrics: which metrics media value should be a top field?. default loss and acc
    Returns:
      dict: results data
    """

    results = {
        "key_metrics": {},
        "idx": self.idx,
        "ts": self.ts,
        "training_size": self.training_size,
        "validation_size": self.validation_size,
        "num_executions": len(self.executions),
        "model": self.model.to_json(),
        "model_name": self.model_name,
        "num_gpu": self.num_gpu,
        "batch_size": self.batch_size,
        "model_size": int(self.model_size),
    }

    # collecting executions results
    exec_metrics = defaultdict(lambda : defaultdict(list))
    executions = [] # execution data
    for execution in self.executions:

      # metrics collection
      for metric, data in execution.metrics.items():
        exec_metrics[metric]['min'].append(execution.metrics[metric]['min'])
        exec_metrics[metric]['max'].append(execution.metrics[metric]['max'])

      # execution data
      execution_info = {
        "num_epochs": execution.num_epochs,
        "history": execution.history,
        "loss_fn": execution.model.loss,
        "loss_weigths": execution.model.loss_weights,
        #FIXME record optimizer parameters
        #"optimizer": execution.model.optimizer
      }
      executions.append(execution_info)

      # save model if needed
      if save_models:
        mdl_base_fname = "%s-%s" % (self.idx, execution.ts)

        # config
        config_fname = "%s-%s-config.json" % (prefix, mdl_base_fname)
        local_path = path.join(local_dir, config_fname)
        with file_io.FileIO(local_path, 'w') as output:
          output.write(execution.model.to_json())
        if gs_dir:
          self.__save_to_gs(config_fname, local_dir, gs_dir)

        # weight
        weights_fname = "%s-%s-weights.h5" % (prefix, mdl_base_fname)
        local_path = path.join(local_dir, weights_fname)
        execution.model.save_weights(local_path)
        if gs_dir:
          self.__save_to_gs(weights_fname, local_dir, gs_dir)


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
    for tm in key_metrics:
      if tm[0] in metrics:
        results['key_metrics'][tm[0]] = metrics[tm[0]][tm[1]]['median']


    fname = '%s-%s-%s-results.json' % (prefix, self.idx, self.training_size)
    output_path = path.join(local_dir, fname)
    with file_io.FileIO(output_path, 'w') as outfile:
      outfile.write(json.dumps(results))
    if gs_dir:
      self.__save_to_gs(fname, local_dir, gs_dir)
    
    self.results = results
    return results
