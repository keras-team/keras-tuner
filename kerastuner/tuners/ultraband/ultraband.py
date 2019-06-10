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

"Ultraband hypertuner"
import copy
import sys
from math import ceil, log

import numpy as np
from tensorflow.keras import backend as K
from termcolor import cprint
from tqdm import tqdm

from kerastuner import config
from kerastuner.abstractions.display import info, subsection, warning, section
from kerastuner.abstractions.io import get_weights_filename
from kerastuner.abstractions.tensorflow import TENSORFLOW as tf
from kerastuner.abstractions.tensorflow import TENSORFLOW_UTILS as tf_utils
from kerastuner.distributions import RandomDistributions
from kerastuner.engine import Tuner

from .ultraband_config import UltraBandConfig


class UltraBand(Tuner):

    def __init__(self, model_fn, objective, **kwargs):
        """ RandomSearch hypertuner
        Args:
            model_fn (function): Function that returns the Keras model to be
            hypertuned. This function is supposed to return a different model
            at every invocation via the use of distribution.* hyperparameters
            range.

            objective (str): Name of the metric to optimize for. The referenced
            metric must be part of the the `compile()` metrics.

        Attributes:
            epoch_budget (int, optional): how many epochs to hypertune for.
            Defaults to 100.

            max_budget (int, optional): how many epochs to spend at most on
            a given model. Defaults to 10.

            min_budget (int, optional): how many epochs to spend at least on
            a given model. Defaults to 3.

            num_executions(int, optional): number of execution for each model.
            Defaults to 1.

            project (str, optional): project the tuning belong to.
            Defaults to 'default'.

            architecture (str, optional): Name of the architecture tuned.
            Default to 'default'.

            user_info(dict, optional): user supplied information that will be
            recorded alongside training data. Defaults to {}.

            label_names (list, optional): Label names for confusion matrix.
            Defaults to None, in which case the numerical labels are used.

            max_model_parameters (int, optional):maximum number of parameters
            allowed for a model. Prevent OOO issue. Defaults to 2500000.

            checkpoint (Bool, optional): Checkpoint model. Setting it to false
            disable it. Defaults to True

            dry_run(bool, optional): Run the tuner without training models.
            Defaults to False.

            debug(bool, optional): Display debug information if true.
            Defaults to False.

            display_model(bool, optional):Display model summary if true.
            Defaults to False.

            results_dir (str, optional): Tuning results dir.
            Defaults to results/. Can specify a gs:// path.

            tmp_dir (str, optional): Temporary dir. Wiped at tuning start.
            Defaults to tmp/. Can specify a gs:// path.

            export_dir (str, optional): Export model dir. Defaults to export/.
            Can specify a gs:// path.

        FIXME:
         - Deal with early stop correctly
         - allows different halving ratio for epochs and models
         - allows differnet type of distribution

        """

        super(UltraBand, self).__init__(model_fn, objective, "UltraBand",
                                        RandomDistributions, **kwargs)

        self.config = UltraBandConfig(kwargs.get('ratio', 3),
                                      self.state.min_epochs,
                                      self.state.max_epochs,
                                      self.state.epoch_budget)

        self.epoch_budget_expensed = 0

        section('UltraBand Tuning')
        subsection('Settings')
        # FIXME use abstraction.display display_settings()
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

        # FIXME: instance should hold model config not model itself as unused
        instance.model = instance.state.recreate_model(
            weights_filename=weights_file)

        # Fit the model
        execution = instance.fit(x, y, **fit_kwargs)
        return execution

    def __train_bracket(self, model_instances, x, y, **fit_kwargs):
        "Train all the models that are in a given bracket."
        if self.state.dry_run:
            return np.random.rand(len(model_instances))

        num_instances = len(model_instances)
        loss_values = []
        for idx, instance in enumerate(model_instances):
            info('Training: %d/%d' % (idx, num_instances))
            execution = self.__train_instance(instance, x, y, **fit_kwargs)
            value = self.__get_sortable_objective_value(execution)
            loss_values.append(value)
        return loss_values

    def search(self, x, y, **kwargs):
        remaining_batches = self.config.num_batches

        while remaining_batches > 0:
            info('Budget: %s/%s - Loop %.2f/%.2f' %
                 (self.epoch_budget_expensed, self.state.epoch_budget,
                  remaining_batches, self.config.num_batches))

            # Last (fractional) loop
            if remaining_batches < 1.0:
                # Reduce the number of models for the last fractional loop
                model_sequence = self.config.partial_batch_epoch_sequence
                if model_sequence is None:
                    break
                info('Partial Batch Model Sequence %s' % model_sequence)
            else:
                model_sequence = self.config.model_sequence

            for band_idx, num_models in enumerate(model_sequence):
                band_total_cost = 0
                info('Budget: %s/%s - Loop %.2f/%.2f - Bands %s/%s' %
                    (self.epoch_budget_expensed, self.state.epoch_budget,
                     remaining_batches, self.config.num_batches, band_idx + 1,
                     self.config.num_bands), 'green')

                num_epochs = self.config.epoch_sequence[band_idx]
                cost = num_models * num_epochs
                self.epoch_budget_expensed += cost
                band_total_cost += cost

                # Generate models
                subsection('|- Generating %s models' % num_models)
                model_instances = []
                kwargs['epochs'] = num_epochs
                if not self.state.dry_run:
                    for _ in tqdm(range(num_models), desc='Generating models',
                                  unit='model'):
                        instance = self.new_instance()
                        if instance is not None:
                            model_instances.append(instance)

                # Training here
                info('Training %s models for %s epochs' % (num_models,
                                                           num_epochs))
                kwargs['epochs'] = int(num_epochs)
                objective_values = self.__train_bracket(model_instances, x, y,
                                                        **kwargs)

                # climbing the band
                brackets = self.config.model_sequence[band_idx + 1:]
                for bracket, num_models in enumerate(brackets):
                    prefix = "--" * bracket
                    num_epochs = self.config.epoch_sequence[bracket + band_idx + 1]
                    cost = num_models * num_epochs
                    self.epoch_budget_expensed += cost
                    band_total_cost += cost

                    # selecting best model
                    band_models = self.__sort_models(model_instances,
                                                     objective_values)
                    cprint("|%sKeeping %d out of %d" %
                           (prefix, num_models, len(band_models)), 'yellow')
                    band_models = band_models[:num_models]  # halve the models

                    # train
                    cprint(
                        '|-%s Training %s models for an additional %s epochs' %
                        (prefix, num_models, num_epochs), 'yellow')
                    kwargs['epochs'] = int(num_epochs)
                    objective_values = self.__train_bracket(
                        band_models, x, y, **kwargs)
            remaining_batches -= 1

    def __get_sortable_objective_value(self, execution):
        metrics = execution.state.metrics
        objective = metrics.get(self.state.objective)
        objective_value = objective.get_last_value()
        if objective.direction == 'max':
            objective_value *= -1
        return objective_value

    def __sort_models(self, models, objective):
        "Return a sorted list of model by loss rate"
        # FIXME remove early stops
        indices = np.argsort(objective)
        sorted_models = []
        for idx in indices:
            sorted_models.append(models[idx])
        return sorted_models
