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

from .instance import Instance
from .logger import Logger
from ..distributions import hyper_parameters


class HyperTuner(object):
    """Abstract hypertuner class."""
    def __init__(self, model_fn, **kwargs):
        self.epoch_budget = kwargs.get('epoch_budget', 3713)
        self.max_epochs = kwargs.get('max_epochs', 50)
        self.min_epochs = kwargs.get('min_epochs', 3)
        self.num_executions = kwargs.get('num_executions', 3) # how many executions
        self.dry_run = kwargs.get('dry_run', False)
        self.max_fail_streak = kwargs.get('max_fail_streak', 20)
        self.num_gpu = kwargs.get('num_gpu', -1)
        self.batch_size = kwargs.get('batch_size', 32)
        self.local_dir = kwargs.get('local_dir', 'results/')
        self.model_name = kwargs.get('model_name', str(int(time.time())))
        self.display_model = kwargs.get('display_model', '') # which models to display
        self.invalid_models = 0 # how many models didn't work
        self.collisions = 0 # how many time we regenerated the same model
        self.instances = {} # All the models we trained
        self.current_instance_idx = -1 # track the current instance trained
        self.model_fn = model_fn
        self.ts = int(time.time())

        #keraslyzer service
        self.gs_dir = None
        if kwargs.get('keraslyzer_user'):
          self.gs_dir = 'gs://keras-tuner.appspot.com/%s/' % kwargs.get('keraslyzer_user')

        #log
        self.log = Logger(self)

        # metrics
        self.METRIC_NAME = 0
        self.METRIC_DIRECTION = 1
        self.max_acc = -1
        self.min_loss = sys.maxsize
        self.max_val_acc = -1
        self.min_val_loss = sys.maxsize


        # including user metrics
        user_metrics = kwargs.get('metrics')
        if user_metrics:
          self.key_metrics = []
          for tm in user_metrics:
            if not isinstance(tm, tuple):
              cprint("[Error] Invalid metric format: %s (%s) - metric format is (metric_name, direction) e.g ('val_acc', 'max') - Ignoring" % (tm, type(tm)), 'red')
              continue
            if tm[self.METRIC_DIRECTION] not in ['min', 'max']:
              cprint("[Error] Invalid metric direction for: %s - metric format is (metric_name, direction). direction is min or max - Ignoring" % tm, 'red')
              continue
            self.key_metrics.append(tm)
        else:
          # sensible default
          self.key_metrics = [('loss', 'min'), ('val_loss', 'min'), ('acc', 'max'), ('val_acc', 'max')]

        # initializing key metrics
        self.stats = {}
        for km in self.key_metrics:
          if km[self.METRIC_DIRECTION] == 'min':
            self.stats[km[self.METRIC_NAME]] = sys.maxsize
          else:
            self.stats[km[self.METRIC_NAME]] = -1

        # output control
        if self.display_model not in ['', 'base', 'multi-gpu', 'both']:
              raise Exception('Invalid display_model value: can be either base, multi-gpu or both')

        # create local dir if needed
        if not os.path.exists(self.local_dir):
          os.makedirs(self.local_dir)
        cprint("|- Saving results in %s" % self.local_dir, 'cyan')

    def get_random_instance(self):
      "Return a never seen before random model instance"
      fail_streak = 0
      while 1:
        fail_streak += 1
        try:
          model = self.model_fn()
        except:
          self.invalid_models += 1
          cprint("[WARN] invalid model %s/%s" % (self.invalid_models, self.max_fail_streak), 'yellow')
          if self.invalid_models >= self.max_fail_streak:
            return None
          continue

        idx = self.__compute_model_id(model)

        if idx not in self.instances:
          break
        self.collisions += 1
      hp = hyper_parameters
      self.instances[idx] = Instance(idx, model, hp, self.model_name, self.num_gpu, self.batch_size, 
                            self.display_model, self.key_metrics, self.local_dir, self.gs_dir)
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
      results = instance.record_results(save_models=save_models)

      #compute overall statisitcs
      for km in self.key_metrics:
        metric_name = km[self.METRIC_NAME]
        if metric_name in results['key_metrics']:
          current_best = self.stats[metric_name]
          res_val = results['key_metrics'][metric_name]
          if km[self.METRIC_DIRECTION] == 'min':
            best = min(current_best, res_val)
          else:
            best = max(current_best, res_val)
          self.stats[metric_name] = best

    def get_model_by_id(self, idx):
      return self.instances.get(idx, None)

    def __compute_model_id(self, model):
      return xxh64(str(model.get_config())).hexdigest()

    def statistics(self):
      #compute overall statisitcs
      latest_instance_results = self.instances[self.current_instance_idx].results
      report = [['Metric', 'Best', 'Last']]
      for km in self.key_metrics:
        metric_name = km[self.METRIC_NAME]
        if metric_name in latest_instance_results['key_metrics']:
          current_best = self.stats[metric_name]
          res_val = latest_instance_results['key_metrics'][metric_name]
          if km[self.METRIC_DIRECTION] == 'min':
            best = min(current_best, res_val)
          else:
            best = max(current_best, res_val)
        report.append([metric_name, best, res_val])
      print (tabulate(report, headers="firstrow"))

      print("Invalid models:%s" % self.invalid_models)
      print("Collisions: %s" % self.collisions)
