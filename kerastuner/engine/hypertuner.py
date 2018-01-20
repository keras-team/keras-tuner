"Meta classs for hypertuner"
import time
import keras
import random
import sys
import json

from termcolor import cprint
from xxhash import xxh32
from tabulate import tabulate

from instance import Instance


class HyperTuner(object):
    """Abstract hypertuner class."""

    def __init__(self, model_fn, **kwargs):
        self.iterations = kwargs.get('iterations', 10) # how many models
        self.executions = kwargs.get('executions', 3) # how many executions
        self.dryrun = kwargs.get('dryrun', False)
        self.max_fail_streak = kwargs.get('max_fail_streak', 20)
        self.invalid_models = 0 # how many models didn't work
        self.collisions = 0 # how many time we regenerated the same model
        self.instances = {} # All the models we trained with their stats and info
        self.model_fn = model_fn
        self.ts = int(time.time())
        #statistics
        self.max_acc = -1
        self.min_loss = sys.maxint
        self.max_val_acc = -1
        self.min_val_loss = sys.maxint
      
    def get_random_instance(self):
      "Return a never seen before random model instance"
      fail_streak = 0
      while 1:
        fail_streak += 1
        try:
          model = self.model_fn()
        except:
          self.invalid_models += 1
          continue

        idx = self.__compute_model_id(model)
        
        if idx not in self.instances:
          break
        self.collisions += 1
        
        if fail_streak == self.max_fail_streak:
          raise Exception('Too many failed models in a row: %s' % fail_streak)


      self.instances[idx] = Instance(model, idx)
      return self.instances[idx] 

    def record_results(self, execution, results):
      "Process the results of an instance training"

      self.min_loss = min(self.min_loss, min(results.history['loss']))
      stats = {'min_loss': self.min_loss} # stats dict is for the progress bar
            
      if 'acc' in results.history:
        self.max_acc = max(self.max_acc, max(results.history['acc']))
        stats['max_acc'] = self.max_acc

      if 'val_acc' in results.history:
        self.max_val_acc = max(self.max_val_acc, max(results.history['val_acc']))
        stats['max_val_acc'] = self.max_val_acc
            
      if 'val_loss' in results.history:
        self.min_val_loss = min(self.min_val_loss, min(results.history['val_loss']))
        stats['min_val_loss'] = self.min_val_loss

      execution.record_results(results)

      return stats

    def get_model_by_id(self, idx):
      return  self.modes.get(idx, None)
      
    def __compute_model_id(self, model):
      return xxh32(str(model.get_config())).hexdigest()

    def statistics(self):
      print "Invalid models:%s" % self.invalid_models
      print "Collisions: %s" % self.collisions
    
    def record_run_info(self, fname):
      "Dump run info into a file for post analysis"

      #FIXME should allows saving to GS as well
      run = {
        'ts': self.ts,
        'iterations': self.iterations,
        'executions': self.executions,
        'min_loss': self.min_loss,
      }

      if self.max_acc != -1:
        run['max_acc'] = self.max_acc
      if self.max_val_acc != -1:
        run['max_val_acc'] = self.max_val_acc
      if self.min_val_loss != sys.maxint:
        run['min_val_loss'] = self.min_val_loss

      instances_stats = []
      for instance in self.instances.values():
        instances_stats.append(instance.get_stats())
      run['instances'] = instances_stats
      with open(fname, 'w') as outfile:
        outfile.write(json.dumps(run))
