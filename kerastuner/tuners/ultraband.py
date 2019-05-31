# Copyright 2019 The Keras Tuner Authors
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     https://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

""""Ultraband hypertuner

Initial algorithm: https://gist.github.com/ebursztein/8304de052a40058fd0ebaf08c949cc1d

"""
import copy
import sys
from math import ceil, log

import numpy as np
from tensorflow.keras import backend as K
from termcolor import cprint
from tqdm import tqdm
from kerastuner.abstractions.tensorflow import TENSORFLOW_UTILS as tf_utils
from kerastuner.abstractions.tensorflow import TENSORFLOW as tf
from kerastuner import config
from kerastuner.abstractions.display import subsection, warning
from kerastuner.abstractions.io import get_weights_filename
from kerastuner.distributions import RandomDistributions
from kerastuner.tuners.ultraband_config import UltraBandConfig

from ..engine import Tuner


class UltraBand(Tuner):
    "UltraBand tuner"

    def __init__(self, model_fn, objective, **kwargs):
        """ UltraBand hypertuner initialization
        Args:
          model_name (str): used to prefix results. Default: ts

          epoch_budget (int): how many epochs to spend on optimization. default 1890
          max_epochs (int): number of epoch to train the best model on. Default 45
          min_epochs (int): minimal number of epoch to train model on. Default 3
          ratio (int): ratio used to grow the distribution. Default 3

          executions (int): number of exection for each model tested

          display_model (str): base: cpu/single gpu version, multi-gpu: multi-gpu, both: base and multi-gpu. default (Nothing)

          num_gpu (int): number of gpu to use. Default 0
          gpu_mem (int): amount of RAM per GPU. Used for batch size calculation

          local_dir (str): where to store results and models. Default results/
          gs_dir (str): Google cloud bucket to use to store results and model (optional). Default None

          dry_run (bool): do not train the model just run the pipeline. Default False
          max_fail_streak (int): number of failed model before giving up. Default 20

        FIXME:
         - Deal with early stop correctly
         - allows different halving ratio for epochs and models
         - allows differnet type of distribution

        """

        super(UltraBand, self).__init__(model_fn, objective, "UltraBand",
                                        RandomDistributions, **kwargs)

        self.ratio = kwargs.get('ratio', 3)

        self.config = UltraBandConfig(
            kwargs.get('ratio', 3),
            self.state.min_epochs,
            self.state.max_epochs,
            self.state.epoch_budget)

        self.epoch_budget_expensed = 0

        cprint('-=[UltraBand Tuning]=-', 'magenta')
        cprint('|- Budget: %s' % self.state.epoch_budget, 'yellow')
        cprint('|- Num models seq %s' % self.config.model_sequence, 'yellow')
        cprint('|- Num epoch seq: %s' %
               self.config.epoch_sequence, 'yellow')
        cprint('|- Bands', 'green')
        cprint('   |- number of bands: %s' %
               len(self.config.model_sequence), 'green')
        cprint('   |- cost per band: %s' %
               self.config.epoch_sequence, 'green')
        cprint('|- Loops', 'blue')
        cprint('   |- number of batches: %s' % self.config.num_batches, 'blue')
        cprint('   |- cost per loop: %s' %
               self.config.epochs_per_batch, 'blue')

    def __get_sortable_objective_value(self, execution):
        metrics = execution.state.metrics
        objective = metrics.get(self.state.objective)
        objective_value = objective.get_last_value()
        if objective.direction == 'max':
            objective_value *= -1
        return objective_value

    def __train_instance(self, instance, x, y, **fit_kwargs):
        tf_utils.clear_tf_session()
        # Determine the weights file (if any) to load, and rebuild the model.
        weights_file = None
        execution = instance.executions.get_last()
        if execution:
            weights_file = get_weights_filename(
                self.state, instance.state, execution.state)
            if not tf.io.gfile.exists(weights_file):
                warning("Could not open weights file: '%s'" % weights_file)
                weights_file = None

        instance.model = instance.state.recreate_model(
            weights_filename=weights_file)

        # Fit the model
        execution = instance.fit(x, y, **fit_kwargs)
        return execution

    def __train_loop(self, model_instances, x, y, **fit_kwargs):
        if self.state.dry_run:
            return np.random.rand(len(model_instances))

        num_instances = len(model_instances)
        loss_values = []
        for idx, instance in enumerate(model_instances):
            cprint('|--- Training %d/%d' % (idx, num_instances), 'green')
            execution = self.__train_instance(instance, x, y, **fit_kwargs)
            value = self.__get_sortable_objective_value(execution)
            loss_values.append(value)
        return loss_values

    def search(self, x, y, **kwargs):
        remaining_batches = self.config.num_batches

        while remaining_batches > 0:
            cprint(
                'Budget: %s/%s - Loop %.2f/%.2f' %
                (self.epoch_budget_expensed, self.state.epoch_budget,
                 remaining_batches, self.config.num_batches), 'blue')

            # Last (fractional) loop
            if remaining_batches < 1.0:
                # Reduce the number of models for the last fractional loop
                model_sequence = self.config.partial_batch_epoch_sequence
                if model_sequence is None:
                    break
                cprint('|- Partial Batch Model Sequence %s' % model_sequence,
                       'yellow')
            else:
                model_sequence = self.config.model_sequence

            for band_idx, num_models in enumerate(model_sequence):
                band_total_cost = 0
                cprint(
                    'Budget: %s/%s - Loop %.2f/%.2f - Bands %s/%s' %
                    (self.epoch_budget_expensed, self.state.epoch_budget,
                     remaining_batches, self.config.num_batches, band_idx + 1,
                     self.config.num_bands), 'green')

                num_epochs = self.config.epoch_sequence[band_idx]
                cost = num_models * num_epochs
                self.epoch_budget_expensed += cost
                band_total_cost += cost

                # Generate models
                cprint('|- Generating %s models' % num_models, 'yellow')
                model_instances = []
                kwargs['epochs'] = num_epochs
                if not self.state.dry_run:
                    for _ in tqdm(range(num_models), desc='Generating models',
                                  unit='model'):
                        instance = self.new_instance()
                        if instance is not None:
                            model_instances.append(instance)


                # Training here
                cprint(
                    '|- Training %s models for %s epochs' %
                    (num_models, num_epochs), 'yellow')
                kwargs['epochs'] = int(num_epochs)
                loss_values = self.__train_loop(
                    model_instances, x, y, **kwargs)

                # climbing the band
                for step, num_models in enumerate(
                        self.config.model_sequence[band_idx + 1:]):
                    prefix = "--" * step
                    num_epochs = self.config.epoch_sequence[step + band_idx + 1]
                    cost = num_models * num_epochs
                    self.epoch_budget_expensed += cost
                    band_total_cost += cost

                    # selecting best model
                    band_models = self.__sort_models(model_instances,
                                                     loss_values)
                    cprint("|%sKeeping %d out of %d" %
                           (prefix, num_models, len(band_models)), 'yellow')
                    band_models = band_models[:num_models]  # halve the models

                    # train
                    cprint(
                        '|-%s Training %s models for an additional %s epochs' %
                        (prefix, num_models, num_epochs), 'yellow')
                    kwargs['epochs'] = int(num_epochs)
                    loss_values = self.__train_loop(
                        band_models, x, y, **kwargs)
            remaining_batches -= 1


    def __sort_models(self, models, loss_values):
        "Return a sorted list of model by loss rate"
        # FIXME remove early stops
        # recall loss is decreasing so use asc
        indices = np.argsort(loss_values)
        sorted_models = []
        for idx in indices:
            sorted_models.append(models[idx])
        return sorted_models
