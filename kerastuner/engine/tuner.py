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
"Keras Tuner class."

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import gc
import os
import traceback


import tensorflow as tf
from tensorflow import keras

from ..abstractions import display
from . import base_tuner
from . import tuner_utils
from .. import config as config_module


class Tuner(base_tuner.BaseTuner):
    """Tuner class for Keras models.

    May be subclassed to create new tuners.

    Args:
        oracle: Instance of Oracle class.
        hypermodel: Instance of HyperModel class
            (or callable that takes hyperparameters
            and returns a Model instance).
        max_model_size: Int. Maximum size of weights
            (in floating point coefficients) for a valid
            models. Models larger than this are rejected.
        optimizer: Optional. Optimizer instance.
            May be used to override the `optimizer`
            argument in the `compile` step for the
            models. If the hypermodel
            does not compile the models it generates,
            then this argument must be specified.
        loss: Optional. May be used to override the `loss`
            argument in the `compile` step for the
            models. If the hypermodel
            does not compile the models it generates,
            then this argument must be specified.
        metrics: Optional. May be used to override the
            `metrics` argument in the `compile` step
            for the models. If the hypermodel
            does not compile the models it generates,
            then this argument must be specified.
        distribution_strategy: Optional. A TensorFlow
            `tf.distribute` DistributionStrategy instance. If
            specified, each trial will run under this scope. For
            example, `tf.distribute.MirroredStrategy(['/gpu:0, /'gpu:1])`
            will run each trial on two GPUs. Currently only
            single-worker strategies are supported.
        directory: String. Path to the working directory (relative).
        project_name: Name to use as prefix for files saved
            by this Tuner.
        logger: Optional. Instance of Logger class, used for streaming data
            to Cloud Service for monitoring.
        tuner_id: Optional. Used only with multi-worker DistributionStrategies.
        reload: Whether an existing project of the same name should be reloaded
            if one is found.
    """

    def __init__(self,
                 oracle,
                 hypermodel,
                 max_model_size=None,
                 optimizer=None,
                 loss=None,
                 metrics=None,
                 distribution_strategy=None,
                 directory=None,
                 project_name=None,
                 logger=None,
                 tuner_id=None,
                 reload=True):
        super(Tuner, self).__init__(oracle=oracle,
                                    hypermodel=hypermodel,
                                    directory=directory,
                                    project_name=project_name,
                                    logger=logger,
                                    tuner_id=tuner_id,
                                    reload=reload)

        # Global search options
        self.max_model_size = max_model_size
        self.distribution_strategy = distribution_strategy

        # Compilation options
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics

        # Private internal state
        self._max_fail_streak = 5
        self._stats = tuner_utils.TunerStats()

        # Save only the last N checkpoints.
        self._save_n_checkpoints = 10

    def run_trial(self, trial, *fit_args, **fit_kwargs):
        # Patch fit arguments. During model `fit`, the patched
        # callbacks call: `self.on_epoch_begin`, `self.on_epoch_end`,
        # `self.on_batch_begin`, `self.on_batch_end`.
        fit_kwargs = copy.copy(fit_kwargs)
        original_callbacks = fit_kwargs.get('callbacks', [])[:]
        fit_kwargs['callbacks'] = self._inject_callbacks(
            original_callbacks, trial)
        model = self._build_model(trial.hyperparameters)
        self._compile_model(model)
        model.fit(*fit_args, **fit_kwargs)

    def save_model(self, trial_id, model, step=0):
        epoch = step
        self._checkpoint_model(model, trial_id, epoch)
        if epoch > self._save_n_checkpoints:
            self._delete_checkpoint(
                trial_id, epoch - self._save_n_checkpoints)

    def load_model(self, trial):
        model = self.hypermodel.build(trial.hyperparameters)
        self._compile_model(model)
        # Reload best checkpoint. The Oracle scores the Trial and also
        # indicates at what epoch the best value of the objective was
        # obtained.
        best_epoch = trial.best_step
        model.load_weights(self._get_checkpoint_fname(
            trial.trial_id, best_epoch))
        return model

    def on_epoch_begin(self, trial, model, epoch, logs=None):
        pass

    def on_batch_begin(self, trial, model, batch, logs):
        pass

    def on_batch_end(self, trial, model, batch, logs=None):
        pass

    def on_epoch_end(self, trial, model, epoch, logs=None):
        self.save_model(trial.trial_id, model, step=epoch)
        # Report intermediate metrics to the `Oracle`.
        status = self.oracle.update_trial(
            trial.trial_id, metrics=logs, step=epoch)
        trial.status = status
        if trial.status == "STOPPED":
            model.stop_training = True

    def get_best_models(self, num_models=1):
        """Returns the best model(s), as determined by the tuner's objective.

        The models are loaded with the weights corresponding to
        their best checkpoint (at the end of the best epoch of best trial).

        This method is only a convenience shortcut.

        Args:
            num_models (int, optional): Number of best models to return.
                Models will be returned in sorted order. Defaults to 1.

        Returns:
            List of trained model instances.
        """
        # Method only exists in this class for the docstring override.
        return super(Tuner, self).get_best_models(num_models)

    def get_state(self):
        state = {'stats': self._stats.get_config()}
        return state

    def set_state(self, state):
        self._stats = tuner_utils.TunerStats.from_config(state['stats'])

    def _build_model(self, hp):
        """Return a never seen before model instance, compiled.

        Args:
            hp: Instance of HyperParameters with populated values.
                These values will be used to instantiate a model
                using the tuner's hypermodel.

        Returns:
            A compiled Model instance.

        Raises:
            RuntimeError: If we cannot generate
                a unique Model instance.
        """
        fail_streak = 0
        oversized_streak = 0

        while 1:
            # clean-up TF graph from previously stored (defunct) graph
            keras.backend.clear_session()
            gc.collect()
            self._stats.num_generated_models += 1
            fail_streak += 1
            try:
                with tuner_utils.maybe_distribute(self.distribution_strategy):
                    model = self.hypermodel.build(hp)
            except:
                if config_module.DEBUG:
                    traceback.print_exc()

                self._stats.num_invalid_models += 1
                display.warning('Invalid model %s/%s' %
                                (self._stats.num_invalid_models,
                                 self._max_fail_streak))

                if self._stats.num_invalid_models >= self._max_fail_streak:
                    raise RuntimeError(
                        'Too many failed attempts to build model.')
                continue

            # Stop if `build()` does not return a valid model.
            if not isinstance(model, keras.models.Model):
                raise RuntimeError(
                    'Model-building function did not return '
                    'a valid Model instance.')

            # Check model size.
            size = tuner_utils.maybe_compute_model_size(model)
            if self.max_model_size and size > self.max_model_size:
                oversized_streak += 1
                self._stats.num_oversized_models += 1
                display.warning(
                    'Oversized model: %s parameters -- skipping' % (size))
                if oversized_streak >= self._max_fail_streak:
                    raise RuntimeError(
                        'Too many consecutive oversized models.')
                continue
            break

        return self._compile_model(model)

    def _compile_model(self, model):
        with tuner_utils.maybe_distribute(self.distribution_strategy):
            if not model.optimizer:
                if not self.optimizer:
                    raise ValueError(
                        'The hypermodel returned a model '
                        'that was not compiled. In this case, you '
                        'should pass the arguments `optimizer`, `loss`, '
                        'and `metrics` to the Tuner constructor, so '
                        'that the Tuner will able to compile the model.')
                model.compile(
                    optimizer=self.optimizer, loss=self.loss, metrics=self.metrics)
            elif self.optimizer or self.loss or self.metrics:
                compile_kwargs = {
                    'optimizer': model.optimizer,
                    'loss': model.loss,
                    'metrics': model.metrics,
                }
                if self.loss:
                    compile_kwargs['loss'] = self.loss
                if self.optimizer:
                    compile_kwargs['optimizer'] = self.optimizer
                if self.metrics:
                    compile_kwargs['metrics'] = self.metrics
                model.compile(**compile_kwargs)
            return model

    def _inject_callbacks(self, callbacks, trial):
        # Deepcopy and patch callbacks if needed
        if callbacks:
            try:
                callbacks = copy.deepcopy(callbacks)
            except:
                raise ValueError(
                    'All callbacks used during a search '
                    'should be deep-copyable (since they are '
                    'reused across trials). '
                    'It is not possible to do `copy.deepcopy(%s)`' %
                    (callbacks,))
            for callback in callbacks:
                # Patching tensorboard log dir
                if callback.__class__.__name__ == 'TensorBoard':
                    callback.log_dir = os.path.join(
                        callback.log_dir,
                        str(trial.trial_id))
        else:
            callbacks = []

        # Add callback to call back the tuner during `fit`.
        callbacks.append(tuner_utils.TunerCallback(self, trial))
        return callbacks

    def _get_checkpoint_dir(self, trial_id, epoch):
        return os.path.join(
            self.get_trial_dir(trial_id),
            'checkpoints',
            'epoch_' + str(epoch))

    def _get_checkpoint_fname(self, trial_id, epoch):
        return os.path.join(
            # Each checkpoint is saved in its own directory.
            self._get_checkpoint_dir(trial_id, epoch),
            'checkpoint')

    def _checkpoint_model(self, model, trial_id, epoch):
        fname = self._get_checkpoint_fname(trial_id, epoch)
        # Save in TF format.
        model.save_weights(fname)
        return fname

    def _delete_checkpoint(self, trial_id, epoch):
        tf.io.gfile.rmtree(self._get_checkpoint_dir(trial_id, epoch))
