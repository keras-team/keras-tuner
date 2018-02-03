"Meta classs for hypertuner"
import time
import keras
import random
import sys
import json
import os
from termcolor import cprint
from xxhash import xxh64 # xxh64 is faster
from tabulate import tabulate

from instance import Instance


class HyperTuner(object):
    """Abstract hypertuner class."""

    def __init__(self, model_fn, **kwargs):
        self.num_iterations = kwargs.get('num_iterations', 10) # how many models
        self.num_executions = kwargs.get('num_executions', 3) # how many executions
        self.dryrun = kwargs.get('dryrun', False)
        self.max_fail_streak = kwargs.get('max_fail_streak', 20)
        self.num_gpu = kwargs.get('num_gpu', -1)
        self.gpu_mem = kwargs.get('gpu_mem', -1)
        self.local_dir = kwargs.get('local_dir', 'results/')
        self.gs_dir = kwargs.get('gs_dir', None)
        self.invalid_models = 0 # how many models didn't work
        self.collisions = 0 # how many time we regenerated the same model
        self.instances = {} # All the models we trained with their stats and info
        self.current_instance_idx = -1 # track the current instance trained
        self.model_fn = model_fn
        self.ts = int(time.time())
        #statistics
        self.max_acc = -1
        self.min_loss = sys.maxint
        self.max_val_acc = -1
        self.min_val_loss = sys.maxint
    
        # create local dir if needed
        if not os.path.exists(self.local_dir):
          os.makedirs(self.local_dir)

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


      self.instances[idx] = Instance(model, idx, self.num_gpu, self.gpu_mem)
      self.current_instance_idx = idx
      return self.instances[idx] 

    def record_results(self, save_models=True, idx=None):
      """Record instance results
      Args:
        save_model (bool): Save the trained models?
        idx (xxhash): index of the instance. By default use the lastest instance for convience.  
      """

      if not idx:
        instance = self.instances[self.current_instance_idx]
      else:
        instance = self.instances[idx]
      instance.record_results(self.local_dir, gs_dir=self.gs_dir, save_models=save_models)

      #FIXME stats here
      #self.min_loss = min(self.min_loss, min(results.history['loss']))
      #stats = {'min_loss': self.min_loss} # stats dict is for the progress bar
            
      #if 'acc' in results.history:
      #  self.max_acc = max(self.max_acc, max(results.history['acc']))
      #  stats['max_acc'] = self.max_acc

      #if 'val_acc' in results.history:
      #  self.max_val_acc = max(self.max_val_acc, max(results.history['val_acc']))
      #  stats['max_val_acc'] = self.max_val_acc
            
      #if 'val_loss' in results.history:
      #  self.min_val_loss = min(self.min_val_loss, min(results.history['val_loss']))
      #  stats['min_val_loss'] = self.min_val_loss

    def get_model_by_id(self, idx):
      return self.modes.get(idx, None)
     
    def __compute_model_id(self, model):
      return xxh64(str(model.get_config())).hexdigest()

    def statistics(self):
      # FIXME expand statistics
      ###       run = {
      ##  'ts': self.ts,
      ##  'iterations': self.num_iterations,
      ##  'executions': self.num_executions,
      ##  'min_loss': self.min_loss,
      ##}

      print "Invalid models:%s" % self.invalid_models
      print "Collisions: %s" % self.collisions