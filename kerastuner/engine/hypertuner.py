"Meta classs for hypertuner"
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from abc import abstractmethod
import time
import sys
import json
import os
from termcolor import cprint
import hashlib

import socket
from tqdm import tqdm
from pathlib import Path
from tensorflow.python.lib.io import file_io  # allows to write to GCP or local
from collections import defaultdict
import tensorflow as tf
import tensorflow.keras.backend as K
import gc

from ..utils import max_model_size
from .display import print_table, section, highlight, subsection
from .backend import Backend
from .instance import Instance
from .logger import Logger
from ..distributions import get_hyper_parameters, clear_hyper_parameters
from ..system import System
from .display import section, setting, warning


class HyperTuner(object):
    """Abstract hypertuner class."""

    def __init__(self, model_fn, tuner_name, **kwargs):
        """
        Args:
            max_params (int): Maximum number of parameters a model can have - anything above will be discarded

            architecture (str): name of the architecture that is being tuned.
            project (str): name of the project the architecture belongs to.

        Notes:
            All architecture meta data are stored into the self.meta_data
            field as they are only used for recording
        """

        # log
        self.log = Logger(self)

        self.epoch_budget = kwargs.get('epoch_budget', 3713)
        self.remaining_budget = self.epoch_budget
        self.max_epochs = kwargs.get('max_epochs', 50)
        self.min_epochs = kwargs.get('min_epochs', 3)

        self.num_executions = kwargs.get('num_executions', 1)
        self.dry_run = kwargs.get('dry_run', False)
        self.debug = kwargs.get('debug', False)
        self.max_fail_streak = kwargs.get('max_fail_streak', 20)
        self.num_gpu = kwargs.get('num_gpu', 0)
        self.batch_size = kwargs.get('batch_size', 32)
        self.display_model = kwargs.get('display_model', '')
        self.display_hyper_parameters = kwargs.get('display_hyper_parameters', False)
        
        # instances management
        self.instances = {}  # All the models we trained
        self.previous_instances = {}  # instance previously trained
        self.current_instance_idx = -1  # track the current instance trained
        self.num_generated_models = 0  # overall number of model generated
        self.num_invalid_models = 0  # how many models didn't work
        self.num_mdl_previously_trained = 0  # how many models already trained
        self.num_collisions = 0  # how many time we regenerated the same model
        self.num_over_sized_models = 0  # num models with params> max_params

        self.model_fn = model_fn
        self.callback_fn = kwargs.get('callback_generator', None)
        self.start_time = int(time.time())
        self.keras_function = 'fit'
        self.info = kwargs.get('info', {})  # additional info provided by users
        self.tuner_name = tuner_name
        self.system = System()
        self.backend = None  # init in the backend() funct if used

        # model checkpointing
        self.checkpoint = {
            "enable": kwargs.get('checkpoint_models', True),
            "metric": kwargs.get('checkpoint_metric', 'val_loss'),
            "mode":  kwargs.get('checkpoint_mode', 'min'),
        }

        if 'accuracy' in self.checkpoint['metric'] and self.checkpoint['mode'] == 'min':
            self.log.warning("Potentially incorrect checkpoint configuration: %s %s  -- change checkpoint_mode to 'max'? "
                             % (self.checkpoint['metric'], self.checkpoint['mode']))

        if 'loss' in self.checkpoint['metric'] and self.checkpoint['mode'] == 'max':
            self.log.warning("Potentially incorrect checkpoint configuration: %s %s  -- change checkpoint_mode to 'max'? "
                             % (self.checkpoint['metric'], self.checkpoint['mode']))

        if self.checkpoint['mode'] != 'min' and self.checkpoint['mode'] != 'max':
            raise Exception(
                'checkpoint_mode must be either min or max - current value:',
                self.checkpoint['mode'])

        if self.checkpoint['enable']:
            self.log.info("Model checkpoint enabled - metric:%s mode:%s" %
                          (self.checkpoint['metric'], self.checkpoint['mode']))

        # Model meta data
        self.meta_data = {
            "architecture": kwargs.get('architecture', str(int(time.time()))),
            "project": kwargs.get('project', 'default'),
            "user_info": self.info
        }

        self.meta_data['server'] = {
            "local_dir": kwargs.get('local_dir', 'results/'),
        }
        self.meta_data['server'].update(self.system.get_status())
        if not self.num_gpu and self.meta_data['server']['available_gpu']:
            # marking gpu 1 as used if TF support gpu
            if self.meta_data['server']['software']['tensorflow_use_gpu']:
                self.num_gpu = 1
            else:
                warning("GPU detected but tensorflow is not compiled to use it")
        self.meta_data['server']['num_gpu'] = self.num_gpu


        # max model size estimation
        max_params = kwargs.get('max_params', 'auto')
        if max_params == 'auto':  # compute max based of our estimate
            if self.meta_data['server']['software']['tensorflow_use_gpu']:
                total = self.meta_data['server']['gpu'][0]['memory']['total']
                used = self.meta_data['server']['gpu'][0]['memory']['used']
                available = total - used
                self.max_params = max_model_size(self.batch_size, available,
                                                 self.num_gpu)
            else:
                total = self.meta_data['server']['ram']['total']
                used = self.meta_data['server']['ram']['used']
                available = total - used
                max_params = max(max_model_size(self.batch_size, available, 1), 5000000)
                self.max_params = max_params
        else:
            self.max_params = max_params

        self.meta_data['tuner'] = {
            "name": self.tuner_name,
            "start_time": self.start_time,

            "epoch_budget": self.epoch_budget,
            "remaining_budget": self.remaining_budget,
            "min_epochs": self.min_epochs,
            "max_epochs": self.max_epochs,
            "max_params": self.max_params,

            "trained_models": self.num_generated_models,
            "collisions": self.num_collisions,
            "invalid_models": self.num_invalid_models,
            "over_size_models": self.num_over_sized_models,
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
                    cprint(
                        "[Error] Invalid metric format: %s (%s) - metric format is (metric_name, direction) e.g ('val_acc', 'max') - Ignoring" % (tm, type(tm)), 'red')
                    continue
                if tm[self.METRIC_DIRECTION] not in ['min', 'max']:
                    cprint(
                        "[Error] Invalid metric direction for: %s - metric format is (metric_name, direction). direction is min or max - Ignoring" % tm, 'red')
                    continue
                self.key_metrics.append(tm)
        else:
            # sensible default
            self.key_metrics = [('loss', 'min'), ('val_loss', 'min'),
                                ('acc', 'max'), ('val_acc', 'max')]

        # initializing stats
        self.stats = {
            'best': {},
            'latest': {},
            'direction': {}
        }
        for km in self.key_metrics:
            self.stats['direction'][km[self.METRIC_NAME]]  = km[self.METRIC_DIRECTION]
            if km[self.METRIC_DIRECTION] == 'min':
                self.stats['best'][km[self.METRIC_NAME]] = sys.maxsize
            else:
                self.stats['best'][km[self.METRIC_NAME]] = -1
        self.meta_data['statistics'] = self.stats

        # output control
        if self.display_model not in ['', 'base', 'multi-gpu', 'both']:
            raise Exception(
                'Invalid display_model value: can be either base, multi-gpu or both')

        # create local dir if needed
        if not os.path.exists(self.meta_data['server']['local_dir']):
            os.makedirs(self.meta_data['server']['local_dir'])
        else:
            # check for models already trained
            self._load_previously_trained_instances(**kwargs)


        # make sure TF session is configured correctly
        cfg = tf.ConfigProto()
        cfg.gpu_options.allow_growth = True
        K.set_session(tf.Session(config=cfg))

        # FIXME: output metadata (move from backend call)

        # recap
        section("Key parameters")
        available_gpu = self.meta_data['server']['available_gpu']
        setting("GPUs Used: %d / %d" % (self.num_gpu, available_gpu), idx=0)
        setting("Model max params: %.1fM" % (self.max_params / 1000000.0), idx=1)
        setting("Saving results in %s" % self.meta_data['server']['local_dir'],
                idx=2)

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
        "Print a summary of the hyperparams search"
        # clean hyperparameters
        clear_hyper_parameters()

        # Calling the model function to get a set of params
        self.model_fn()
        hyper_parameters = get_hyper_parameters()

        # if no hyper_params returning
        if len(hyper_parameters) == 0:
            self.log.info("No hyper-parameters")
            return

        # Compute the size of the hyperparam space by generating a model
        group_size = defaultdict(lambda: 1)
        total_size = 1
        table = [['Group', 'Param', 'Space size']]

        # param by param
        for name, data in hyper_parameters.items():
            row = [data['group'], data['name'], data['space_size']]
            table.append(row)
            group_size[data['group']] *= data['space_size']
            total_size *= data['space_size']

        # by group
        group_table = [['Group', 'Size']]
        for g, v in group_size.items():
            group_table.append([g, v])

        section("Hyper-parmeters search space")
        subsection("Search space by parameters group")
        print_table(group_table)
        subsection("Search space for each parameters")
        print_table(table)
        highlight("Total search space:%s" % total_size)

    def enable_cloud(self, api_key, **kwargs):
        """Setup backend configuration

            Args:
                api_key (str): The backend API access token.
                kwargs (dict): Optional. Contains the key "url", pointing to the
                base url of the backend.
            Note: this is called by the user
        """
        self.backend = Backend(
            api_key=api_key,
            url=kwargs.get('url', 'https://us-central1-keras-tuner.cloudfunctions.net/api'),
            notifications={
                "crash": kwargs.get("crash_notification", False),
                "tuning_completion": kwargs.get("tuning_completion_notification", False),
                "instance_completion": kwargs.get("instance_completion_notification", False)
            }
        )

        #! fixe this metadata should NOT BE tied to backend setup
        #fname = '%s-%s-meta_data.json' % (self.meta_data['project'],
        #                                  self.meta_data['architecture'])
        #local_path = os.path.join(self.meta_data['server']['local_dir'], fname)
        #with file_io.FileIO(local_path, 'w') as output:
        #    output.write(json.dumps(self.meta_data))
        #backend.cloud_save(local_path=local_path,
        #                   ftype='meta_data', meta_data=self.meta_data)

    def search(self, x, y, **kwargs):
        self.keras_functionkera_function = 'fit'
        self.hypertune(x, y, **kwargs)
        if self.backend:
          self.backend.quit()

    def search_generator(self, x, **kwargs):
        self.keras_function = 'fit_generator'
        y = None  # fit_generator don't use this so we put none to be able to have a single hypertune function
        self.hypertune(x, y, **kwargs)
        if self.backend:
          self.backend.quit()

    def _clear_tf_graph(self):
        """ Clear the content of the TF graph to ensure
            we have a valid model is in memory
        """
        K.clear_session()
        gc.collect()

    def get_random_instance(self):
        "Return a never seen before random model instance"
        fail_streak = 0
        collision_streak = 0
        over_sized_streak = 0

        while 1:

            # clear the hyperparmaters table between run
            clear_hyper_parameters()

            # clean-up TF graph from previously stored (defunct) graph
            self._clear_tf_graph()
            self.num_generated_models += 1
            fail_streak += 1
            try:
                model = self.model_fn()
            except:
                if self.debug:
                    import traceback
                    traceback.print_exc()

                self.num_invalid_models += 1
                self.log.warning("invalid model %s/%s" %
                                 (self.num_invalid_models,
                                  self.max_fail_streak))

                if self.num_invalid_models >= self.max_fail_streak:
                    return None
                continue

            # stop if the model_fn() return nothing
            if not model:
                return None

            idx = self.__compute_model_id(model)

            if idx in self.previous_instances:
                self.log.info("model %s already trained -- skipping" % idx)
                self.num_mdl_previously_trained += 1
                continue

            if idx in self.instances:
                collision_streak += 1
                self.num_collisions += 1
                self.meta_data['tuner']['collisions'] = self.num_collisions
                self.log.warning(
                    "collision detected - model %s already trained -- skipping" % (idx))
                if collision_streak >= self.max_fail_streak:
                    return None
                continue
            hyper_parameters = get_hyper_parameters()
            self._update_metadata()
            instance = Instance(idx, model, hyper_parameters, self.meta_data, self.num_gpu, self.batch_size,
                                self.display_model, self.key_metrics, self.keras_function, self.checkpoint,
                                self.callback_fn, self.backend, self.display_hyper_parameters)
            num_params = instance.compute_model_size()
            if num_params > self.max_params:
                over_sized_streak += 1
                self.num_over_sized_models += 1
                self.log.warning(
                    "Oversized model: %s parameters-- skipping" % (num_params))
                if over_sized_streak >= self.max_fail_streak:
                    return None
                continue

            break

        self.instances[idx] = instance
        self.current_instance_idx = idx
        self.log.new_instance(instance, self.num_generated_models,
                              self.remaining_budget)
        return self.instances[idx]

    def record_results(self, idx=None):
        """Record instance results
        Args:
          idx (str): index of the instance. (default last trained)
        """

        if not idx:
            instance = self.instances[self.current_instance_idx]
        else:
            instance = self.instances[idx]

        results = instance.record_results()

        # compute overall statistics
        latest_results = {}
        best_results = {}
        for km in self.key_metrics:
            metric_name = km[self.METRIC_NAME]
            if metric_name in results['key_metrics']:
                current_best = self.stats['best'][metric_name]
                res_val = results['key_metrics'][metric_name]
                latest_results[metric_name] = res_val
                if km[self.METRIC_DIRECTION] == 'min':
                    best_results[metric_name] = min(current_best, res_val)
                else:
                    best_results[metric_name] = max(current_best, res_val)

        # updating
        self.stats['best'] = best_results
        self.stats['latest'] = latest_results
        self.meta_data['statistics'] = self.stats

    def done(self):
        self.log.done()

    def get_model_by_id(self, idx):
        return self.instances.get(idx, None)

    def __compute_model_id(self, model):
        "compute model hash"
        s = str(model.get_config())
        return hashlib.sha256(s.encode('utf-8')).hexdigest()[:32]

    def _update_metadata(self):
        "update metadata with latest hypertuner state"

        md = self.meta_data['tuner']
        md['remaining_budget'] = self.remaining_budget
        # stats are updated at instance selection not training end
        md['trained_models'] = self.num_generated_models
        md['collisions'] = self.num_collisions
        md['invalid_models'] = self.num_invalid_models
        md['over_size_models'] = self.num_over_sized_models

    @abstractmethod
    def hypertune(self, x, y, **kwargs):
        "method called by the hypertuner to train an instance"
