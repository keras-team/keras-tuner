"Meta classs for hypertuner"
from abc import abstractmethod
import time
import tensorflow.keras as keras
import random
import sys
import json
import os
from termcolor import cprint
import farmhash
from tabulate import tabulate
import socket
from tqdm import tqdm
from pathlib import Path
from tensorflow.python.lib.io import file_io # allows to write to GCP or local
from collections import defaultdict
import tensorflow as tf
import tensorflow.keras.backend as K

from . import backend
from .instance import Instance
from .logger import Logger
from ..distributions import hyper_parameters

class HyperTuner(object):
    """Abstract hypertuner class."""
    def __init__(self, model_fn, **kwargs):
        """
        Args:
            max_params (int): Maximum number of parameters a model can have - anything above will be discarded

            architecture (str): name of the architecture that is being tuned.
            project (str): name of the project the architecture belongs to.

        Notes:
            All architecture meta data are stored into the self.meta_data field as they are only used for recording
        """

        #log
        self.log = Logger(self)

        self.epoch_budget = kwargs.get('epoch_budget', 3713)
        self.max_epochs = kwargs.get('max_epochs', 50)
        self.min_epochs = kwargs.get('min_epochs', 3)
        self.num_executions = kwargs.get('num_executions', 1) # how many executions
        self.dry_run = kwargs.get('dry_run', False)
        self.max_fail_streak = kwargs.get('max_fail_streak', 20)
        self.num_gpu = kwargs.get('num_gpu', 0)
        self.batch_size = kwargs.get('batch_size', 32)
        self.max_params = kwargs.get('max_params', 100000000)

        self.display_model = kwargs.get('display_model', '') # which models to display

        # instances management 
        self.instances = {} # All the models we trained
        self.previous_instances = {} # instance previously trained 
        self.current_instance_idx = -1 # track the current instance trained
        self.invalid_models = 0 # how many models didn't work
        self.skipped_models = 0 # how many models were already trained 
        self.collisions = 0 # how many time we regenerated the same model
        self.over_sized_models = 0 # how many models had a param counts > max_params

        self.model_fn = model_fn
        self.callback_fn = kwargs.get('callback_generator', None)
        self.ts = int(time.time())
        self.keras_function = 'fit'
        self.info = kwargs.get('info', {}) # additional info provided by users
        self.tuner_name = 'default'
        #model checkpointing
        self.checkpoint = {
          "enable": kwargs.get('checkpoint_models', True),
          "metric": kwargs.get('checkpoint_metric', 'val_loss'),
          "mode":  kwargs.get('checkpoint_mode', 'min'),
        }
          
        if self.checkpoint['mode'] != 'min' and self.checkpoint['mode'] != 'max':
          raise Exception('checkpoint_mode must be either min or max - current value:', self.checkpoint['mode'])

        if self.checkpoint['enable']:
          self.log.info("Model checkpoint enabled - metric:%s mode:%s" % (self.checkpoint['metric'], self.checkpoint['mode']))

        # Model meta data
        self.meta_data = {
            "architecture": kwargs.get('architecture', str(int(time.time()))),
            "project": kwargs.get('project', 'default'),
            "user_info": self.info
        }

        self.meta_data['server'] = {
                "local_dir": kwargs.get('local_dir', 'results/'),
                "hostname": socket.gethostname(),
                "num_gpu": self.num_gpu,
            }

        self.meta_data['tuner'] = {
            "name": self.tuner_name,
            "epoch_budget": self.epoch_budget,
            "min_epochs": self.min_epochs,
            "max_epochs": self.max_epochs,
            "max_params": self.max_params,
            "collisions": self.collisions,
            "over_size_models": self.over_sized_models,
            "checkpoint": self.checkpoint
            }

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
        if not os.path.exists(self.meta_data['server']['local_dir']):
          os.makedirs(self.meta_data['server']['local_dir'])
        else:
          #check for models already trained
          self._load_previously_trained_instances(**kwargs)
        cprint("|- Saving results in %s" % self.meta_data['server']['local_dir'], 'cyan') #fixme use logger

        # make sure TF session is configured correctly
        cfg = tf.ConfigProto()
        cfg.gpu_options.allow_growth = True
        K.set_session(tf.Session(config=cfg))

    def set_tuner_name(self, name):
      self.tuner_name = name
      self.log.tuner_name(self.tuner_name)


    def _load_previously_trained_instances(self, **kwargs):
      "Checking for existing models"
      result_path = Path(kwargs.get('local_dir', 'results/'))
      filenames = list(result_path.glob('*-results.json'))
      for filename in tqdm(filenames, unit='model', desc='Finding previously trained models'):
        data = json.loads(open(str(filename)).read())

        # Narrow down to matching project and architecture
        if (data['meta_data']['architecture'] == self.meta_data['architecture'] 
            and data['meta_data']['project'] == self.meta_data['project']):
              # storing previous instance results in memory in case the tuner needs them.
              self.previous_instances[data['meta_data']['instance']] = data

    def summary(self):
      global hyper_parameters
  
      #compute the size of the hyperparam space by generating a model
      group_size = defaultdict(lambda:1)
      total_size = 1
      table = [['Group', 'Param', 'Space size']]
      
      #param by param
      self.log.section("Hyper-params search space by params")
      for name, data in hyper_parameters.items():
        row = [data['group'], name, data['space_size']]
        table.append(row)
        group_size[data['group']] *= data['space_size']
        total_size *= data['space_size']
      self.log.text(tabulate(table, headers="firstrow", tablefmt="grid"))
      
      #by group
      self.log.section("Hyper-params search space by group")
      group_table = [['Group', 'Size']]
      for g, v in group_size.items():
        group_table.append([g, v])
      self.log.text(tabulate(group_table, headers="firstrow", tablefmt="grid"))

      self.log.text("Total search space:%s" % total_size)
      


    def backend(self, username, **kwargs):
        """Setup backend configuration
        
          Args
            info (dict): free form dictionary of information supplied by the user about the hypertuning. MUST be JSON serializable.
        """

        self.meta_data['backend']  = {
            "username": username,
            "url": kwargs.get('url', 'gs://keras-tuner.appspot.com/'),
            "crash_notification": kwargs.get("crash_notification", False),
            "tuning_completion_notification": kwargs.get("tuning_completion_notification", False),
            "instance_trained_notification": kwargs.get("instance_trained_notification", False),
        }

        config_fname = '%s-%s-meta_data.json' % (self.meta_data['project'], self.meta_data['architecture'])
        local_path = os.path.join(self.meta_data['server']['local_dir'], config_fname)
        with file_io.FileIO(local_path, 'w') as output:
            output.write(json.dumps(self.meta_data))
        backend.cloud_save(local_path=local_path, ftype='meta_data', meta_data=self.meta_data)


    def search(self, x, y, **kwargs):
        self.keras_functionkera_function = 'fit'
        self.hypertune(x, y, **kwargs)
    
    def search_generator(self, x, **kwargs):
        self.keras_function = 'fit_generator'
        y = None # fit_generator don't use this so we put none to be able to have a single hypertune function
        self.hypertune(x, y, **kwargs)

    def get_random_instance(self):
      "Return a never seen before random model instance"
      global hyper_parameters
      fail_streak = 0
      collision_streak = 0
      over_sized_streak = 0
      while 1:
        fail_streak += 1
        try:
          model = self.model_fn()
        except:
          self.invalid_models += 1
          self.log.warning("invalid model %s/%s" % (self.invalid_models, self.max_fail_streak))
          if self.invalid_models >= self.max_fail_streak:
            return None
          continue
        
        # stop if the model_fn() return nothing
        if not model:
          return None

        idx = self.__compute_model_id(model)

        if idx in self.previous_instances:
          self.log.info("model %s already trained -- skipping" % idx)
          self.skipped_models += 1
          continue

        if idx in self.instances:
          collision_streak += 1
          self.collisions += 1
          self.meta_data['tuner']['collisions'] = self.collisions
          self.log.warning("collision detect model %s already trained -- skipping" % (idx))
          if collision_streak >= self.max_fail_streak:
            return None
          continue

        instance = Instance(idx, model, hyper_parameters, self.meta_data, self.num_gpu, self.batch_size, 
                            self.display_model, self.key_metrics, self.keras_function, self.checkpoint, self.callback_fn)
        num_params = instance.compute_model_size()
        if num_params > self.max_params:
          over_sized_streak += 1
          self.over_sized_models += 1
          self.meta_data['tuner']['over_sized_model'] = self.over_sized_models
          cprint("[WARN] Oversized model: %s parameters-- skipping" % (num_params), 'yellow')
          if over_sized_streak >= self.max_fail_streak:
            return None
          continue
        
        break

      self.instances[idx] = instance
      self.current_instance_idx = idx
      return self.instances[idx]

    def record_results(self, idx=None):
      """Record instance results
      Args:
        idx (str): index of the instance. By default use the lastest instance for convenience.
      """

      if not idx:
        instance = self.instances[self.current_instance_idx]
      else:
        instance = self.instances[idx]
      results = instance.record_results()

      #compute overall statistics
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
      return hex(farmhash.hash64(str(model.get_config())))[2:]  #remove the 0x

    def statistics(self):
      #compute overall statistics
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

      print("Invalid models:\t%s" % self.invalid_models)
      print("Collisions:\t%s" % self.collisions)
      print("Skipped models:\t%s" % self.skipped_models)
    
    @abstractmethod
    def hypertune(self, x, y, **kwargs):
      "method called by the hypertuner to train an instance"
