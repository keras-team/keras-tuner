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

"Tuner that runs multiple executions per Trial."

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from . import tuner as tuner_module
from . import tuner_utils

import collections
import copy
import numpy as np
import os
from tensorflow import keras


class MultiExecutionTuner(tuner_module.Tuner):
    """A Tuner class that averages multiple runs of the process.

    Args:
        oracle: Instance of Oracle class.
        hypermodel: Instance of HyperModel class
            (or callable that takes hyperparameters
            and returns a Model instance).
        executions_per_trial: Int. Number of executions
            (training a model from scratch,
            starting from a new initialization)
            to run per trial (model configuration).
            Model metrics may vary greatly depending
            on random initialization, hence it is
            often a good idea to run several executions
            per trial in order to evaluate the performance
            of a given set of hyperparameter values.
        **kwargs: Keyword arguments relevant to all `Tuner` subclasses.
            Please see the docstring for `Tuner`.
    """

    def __init__(self,
                 oracle,
                 hypermodel,
                 executions_per_trial=1,
                 **kwargs):
        super(MultiExecutionTuner, self).__init__(
            oracle, hypermodel, **kwargs)
        if isinstance(oracle.objective, list):
            raise ValueError(
                'Multi-objective is not supported, found: {}'.format(
                    oracle.objective))
        self.executions_per_trial = executions_per_trial
        # This is the `step` that will be reported to the Oracle at the end
        # of the Trial. Since intermediate results are not used, this is set
        # to 0.
        self._reported_step = 0

    def on_epoch_end(self, trial, model, epoch, logs=None):
        # Intermediate results are not passed to the Oracle, and
        # checkpointing is handled via a `ModelCheckpoint` callback.
        pass

    def run_trial(self, trial, *fit_args, **fit_kwargs):
        model_checkpoint = keras.callbacks.ModelCheckpoint(
            filepath=self._get_checkpoint_fname(
                trial.trial_id, self._reported_step),
            monitor=self.oracle.objective.name,
            mode=self.oracle.objective.direction,
            save_best_only=True,
            save_weights_only=True)
        original_callbacks = fit_kwargs.pop('callbacks', [])

        # Run the training process multiple times.
        metrics = collections.defaultdict(list)
        for execution in range(self.executions_per_trial):
            copied_fit_kwargs = copy.copy(fit_kwargs)
            callbacks = self._deepcopy_callbacks(original_callbacks)
            self._configure_tensorboard_dir(callbacks, trial.trial_id, execution)
            callbacks.append(tuner_utils.TunerCallback(self, trial))
            # Only checkpoint the best epoch across all executions.
            callbacks.append(model_checkpoint)
            copied_fit_kwargs['callbacks'] = callbacks

            model = self.hypermodel.build(trial.hyperparameters)
            self._on_train_begin(model, trial.hyperparameters,
                                 *fit_args, **copied_fit_kwargs)
            history = model.fit(*fit_args, **copied_fit_kwargs)
            for metric, epoch_values in history.history.items():
                if self.oracle.objective.direction == 'min':
                    best_value = np.min(epoch_values)
                else:
                    best_value = np.max(epoch_values)
                metrics[metric].append(best_value)

        # Average the results across executions and send to the Oracle.
        averaged_metrics = {}
        for metric, execution_values in metrics.items():
            averaged_metrics[metric] = np.mean(execution_values)
        self.oracle.update_trial(
            trial.trial_id, metrics=averaged_metrics, step=self._reported_step)

    def _configure_tensorboard_dir(self, callbacks, trial_id, execution=0):
        for callback in callbacks:
            # Patching tensorboard log dir
            if callback.__class__.__name__ == 'TensorBoard':
                callback.log_dir = os.path.join(
                    str(callback.log_dir),
                    trial_id,
                    'execution{}'.format(execution))
        return callbacks
