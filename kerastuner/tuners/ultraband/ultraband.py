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
from tqdm import tqdm

from kerastuner import config
from kerastuner.abstractions.display import info, subsection, warning, section
from kerastuner.abstractions.display import display_settings
from kerastuner.abstractions.io import get_weights_filename
from kerastuner.abstractions.tensorflow import TENSORFLOW as tf
from kerastuner.abstractions.tensorflow import TENSORFLOW_UTILS as tf_utils
from kerastuner.collections import InstanceStatesCollection
from kerastuner.distributions import RandomDistributions
from kerastuner.engine import Tuner
from kerastuner.engine.instance import Instance

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

        self.config = UltraBandConfig(kwargs.get('ratio',
                                                 3), self.state.min_epochs,
                                      self.state.max_epochs,
                                      self.state.epoch_budget)

        self.epoch_budget_expensed = 0

        settings = {
            "Epoch Budget": self.state.epoch_budget,
            "Num Models Sequence": self.config.model_sequence,
            "Num Epochs Sequence": self.config.epoch_sequence,
            "Num Brackets": self.config.num_brackets,        
            "Number of Iterations": self.config.num_batches,
            "Total Cost per Band": self.config.total_epochs_per_band
        }

        section('UltraBand Tuning')
        subsection('Settings')                
        display_settings(settings)

    def __load_instance(self, instance_state):
        # Determine the weights file (if any) to load, and rebuild the model.
        weights_file = None

        if instance_state.execution_states_collection:
            esc = instance_state.execution_states_collection
            execution_state = esc.get_last()
            weights_file = get_weights_filename(self.state, instance_state,
                                                execution_state)
            if not tf.io.gfile.exists(weights_file):
                warning("Could not open weights file: '%s'" % weights_file)
                weights_file = None

        model = instance_state.recreate_model(weights_filename=weights_file)

        return Instance(instance_state.idx,
                        model,
                        instance_state.hyper_parameters,
                        self.state,
                        self.cloudservice,
                        instance_state=instance_state)

    def __train_instance(self, instance, x, y, **fit_kwargs):
        tf_utils.clear_tf_session()

        # Reload the Instance
        instance = self.__load_instance(instance)

        # Fit the model
        instance.fit(x, y, **fit_kwargs)

    def __train_bracket(self, instance_collection, num_epochs, x, y,
                        **fit_kwargs):
        "Train all the models that are in a given bracket."
        num_instances = len(instance_collection)

        info('Training %d models for %d epochs.' % (num_instances, num_epochs))
        for idx, instance in enumerate(instance_collection.to_list()):
            info('  Training: %d/%d' % (idx, num_instances))
            self.__train_instance(instance,
                                  x,
                                  y,
                                  epochs=num_epochs,
                                  **fit_kwargs)

    def __filter_early_stops(self, instance_collection, epoch_target):
        filtered_instances = []
        for instance in instance_collection:
            last_execution = instance.execution_states_collection.get_last()
            if not last_execution.metrics or not last_execution.metrics.exist(
                    "loss"):
                info("Skipping instance %s - no metrics." % instance.idx)
                continue
            metric = last_execution.metrics.get("loss")
            epoch_history_len = len(metric.history)
            if epoch_history_len < epoch_target:
                info("Skipping instance %s - history is only %d epochs long - "
                     "expected %d - assuming early stop." %
                     (instance.idx, epoch_history_len, epoch_target))
                continue

            filtered_instances.append(instance)
        return filtered_instances

    def __bracket(self, instance_collection, num_to_keep, num_epochs,
                  total_num_epochs, x, y, **fit_kwargs):
        self.__train_bracket(instance_collection, num_epochs, x, y,
                             **fit_kwargs)
        instances = instance_collection.sort_by_objective()
        instances = self.__filter_early_stops(instances, total_num_epochs)

        if len(instances) > num_to_keep:
            instances = instances[:num_to_keep]
            info("Keeping %d instances out of %d" %
                 (len(instances), len(instance_collection)))

        output_collection = InstanceStatesCollection()
        for instance in instances:
            output_collection.add(instance.idx, instance)
        return output_collection

    def search(self, x, y, **kwargs):
        assert 'epochs' not in kwargs, \
            "Number of epochs is controlled by the tuner."
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

            # Generate N models, and perform the initial training.
            subsection('Generating %s models' % model_sequence[0])
            candidates = InstanceStatesCollection()
            num_models = self.config.model_sequence[0]

            if not self.state.dry_run:
                for _ in tqdm(range(num_models),
                              desc='Generating models',
                              unit='model'):
                    instance = self.new_instance()
                    if instance is not None:
                        candidates.add(instance.state.idx, instance.state)

            if not candidates:
                info("No models were generated.")
                break

            subsection("Training models.")

            for bracket_idx, num_models in enumerate(model_sequence):
                num_to_keep = 0
                if bracket_idx < len(model_sequence) - 1:
                    num_to_keep = model_sequence[bracket_idx + 1]
                    info("Running a bracket to reduce from %d to %d models" %
                         (num_models, num_to_keep))
                else:
                    info("Running final bracket.")

                info('Budget: %s/%s - Loop %.2f/%.2f - Brackets %s/%s' %
                     (self.epoch_budget_expensed, self.state.epoch_budget,
                      remaining_batches, self.config.num_batches,
                      bracket_idx + 1, self.config.num_brackets))

                num_epochs = self.config.delta_epoch_sequence[bracket_idx]
                total_num_epochs = self.config.epoch_sequence[bracket_idx]
                self.epoch_budget_expensed += num_models * num_epochs

                candidates = self.__bracket(candidates, num_to_keep,
                                            num_epochs, total_num_epochs, x, y,
                                            **kwargs)

            remaining_batches -= 1
