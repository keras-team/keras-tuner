"Meta classs for hypertuner"
import keras
import random
import copy
import sys

from termcolor import cprint
from xxhash import xxh32
from tabulate import tabulate

from instance import Instance


class HyperTuner(object):
    """Abstract hypertuner class."""

    def __init__(self, model_fn, **kwargs):
        self.iterations = kwargs.get('iterations', 10)
        self.runs = kwargs.get('runs', 3)
        self.dryrun = kwargs.get('dryrun', False)
        self.max_fail_streak = kwargs.get('max_fail_streak', 20)
        self.invalid_models = 0 # how many models didn't work
        self.collisions = 0 # how many time we regenerated the same model
        self.instances = {} # All the models we trained with their stats and info
        self.model_fn = model_fn

        #statistics
        self.max_acc = 0
        self.min_loss = sys.maxint
        self.max_val_acc = 0
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

    def record_results(self, instance, results):
      "Process the results of an instance training"
      self.max_acc = max(self.max_acc, results.history['acc'][-1])
      self.min_loss = min(self.min_loss, results.history['loss'][-1])

      #for progress bar
      stats = {'max_acc': self.max_acc, 'min_loss': self.min_loss}
            
      if 'val_acc' in results.history:
        self.max_val_acc = max(self.max_val_acc, results.history['val_acc'][-1])
        stats['max_val_acc'] = self.max_val_acc
            
      if 'val_loss' in results.history:
        self.min_val_loss = min(self.min_val_loss, results.history['val_loss'][-1])
        stats['min_val_loss'] = self.min_val_loss

      return stats

    def get_model_by_id(self, idx):
      return  self.modes.get(idx, None)
      
    def __compute_model_id(self, model):
      return xxh32(str(model.get_config())).hexdigest()

    def statistics(self):
      print "Invalid models:%s" % self.invalid_models
      print "Collisions: %s" % self.collisions
      
