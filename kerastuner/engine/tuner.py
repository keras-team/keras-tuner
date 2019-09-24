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
"Tuner base class."

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import json
import os
import time
import traceback


import numpy as np
import tensorflow as tf
from tensorflow import keras

from kerastuner.engine import stateful
from . import hyperparameters as hp_module
from . import hypermodel as hm_module
from . import oracle as oracle_module
from . import trial as trial_module
from . import tuner_utils
from .. import config as config_module
from .. import utils
from ..abstractions import display
from ..abstractions.tensorflow import TENSORFLOW_UTILS as tf_utils
from . import metrics_tracking


class Tuner(stateful.Stateful):
    """Tuner base class.

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
    """

    def __init__(self,
                 oracle,
                 hypermodel,
                 executions_per_trial=1,
                 max_model_size=None,
                 optimizer=None,
                 loss=None,
                 metrics=None,
                 distribution_strategy=None,
                 directory=None,
                 project_name=None,
                 logger=None,
                 tuner_id=None):
        if not isinstance(oracle, oracle_module.Oracle):
            raise ValueError('Expected oracle to be '
                             'an instance of Oracle, got: %s' % (oracle,))
        self.oracle = oracle

        if isinstance(hypermodel, hm_module.HyperModel):
            self.hypermodel = hypermodel
        else:
            if not callable(hypermodel):
                raise ValueError(
                    'The `hypermodel` argument should be either '
                    'a callable with signature `build(hp)` returning a model, '
                    'or an instance of `HyperModel`.')
            self.hypermodel = hm_module.DefaultHyperModel(hypermodel)

        # Global search options
        self.max_model_size = max_model_size
        self.distribution_strategy = distribution_strategy
        self.tuner_id = tuner_id if tuner_id is not None else 0

        # Compilation options
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics

        # Ops and metadata
        self.directory = directory or '.'
        self.project_name = project_name or 'untitled_project'

        # Private internal state
        self._max_fail_streak = 5
        self._stats = tuner_utils.TunerStats()

        # Logs etc
        self.logger = logger
        self._display = tuner_utils.Display()

        # Populate initial search space.
        if self.oracle.tune_new_entries:
            hp = self.oracle.get_space()
            self._build_model(hp)
            self.oracle.update_space(hp)

    def search(self, *fit_args, **fit_kwargs):
        self.on_search_begin()
        while True:
            trial = self.oracle.create_trial(self.tuner_id)
            if trial.status == trial_module.TrialStatus.STOPPED:
                # Oracle triggered exit.
                break
            if trial.status == trial_module.TrialStatus.IDLE:
                # Oracle is calculating, resend request.
                continue

            self.on_trial_begin(trial)
            self.run_trial(trial, *fit_args, **fit_kwargs)
            self.on_trial_end(trial)
        self.on_search_end()

    def run_trial(self, trial, *fit_args, **fit_kwargs):
        # Patch fit arguments. During model `fit`, the patched 
        # callbacks call: `self.on_epoch_begin`, `self.on_epoch_end`,
        # `self.on_batch_begin`, `self.on_batch_end`.
        fit_kwargs = copy.copy(fit_kwargs)
        original_callbacks = fit_kwargs.get('callbacks', [])[:]
        fit_kwargs['callbacks'] = self._inject_callbacks(
            original_callbacks, trial)
        model = self._build_model(trial.hyperparameters.copy())
        self._compile_model(model)
        return model.fit(*fit_args, **fit_kwargs)

    def on_search_begin(self):
        if self.logger:
            self.logger.register_tuner(self.get_state())

    def on_trial_begin(self, trial):
        if self.logger:
            self.logger.register_trial(trial.trial_id, trial.get_state())

    def on_epoch_begin(self, trial, model, epoch, logs=None):
        pass

    def on_batch_begin(self, trial, model, batch, logs):
        pass

    def on_batch_end(self, trial, model, batch, logs=None):
        pass

    def on_epoch_end(self, trial, model, epoch, logs=None):
        # TODO: Garbage collect unneeded checkpoints.
        self._checkpoint_model(model, trial, epoch)

        # Report intermediate metrics to the `Oracle`.
        status = self.oracle.update_trial(
            trial.trial_id, metrics=logs, step=epoch)
        trial.status = status
        if trial.status == "STOPPED":
            model.stop_training = True

    def on_trial_end(self, trial):
        self.oracle.end_trial(
            trial.trial_id, trial_module.TrialStatus.COMPLETED)
        self._checkpoint_trial(trial)
        self._display.on_trial_end(trial)
        self.save()

    def on_search_end(self):
        if self.logger:
            self.logger.exit()

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
        best_trials = self.oracle.get_best_trials(num_models)
        models = []
        for trial in best_trials:
            hp = trial.hyperparameters.copy()
            model = self.hypermodel.build(hp)
            self._compile_model(model)
            # Reload best checkpoint. The Oracle scores the Trial and also
            # indicates at what epoch the best value of the objective was
            # obtained.
            best_epoch = trial.best_step
            model.load_weights(self._get_checkpoint_fname(trial, best_epoch))
            models.append(model)
        return models

    def search_space_summary(self, extended=False):
        """Print search space summary.

        Args:
            extended: Bool, optional. Display extended summary.
                Defaults to False.
        """
        display.section('Search space summary')
        hp = self.oracle.get_space()
        display.display_setting(
            'Default search space size: %d' % len(hp.space))
        for p in hp.space:
            config = p.get_config()
            name = config.pop('name')
            display.subsection('%s (%s)' % (name, p.__class__.__name__))
            display.display_settings(config)

    def results_summary(self, num_trials=10):
        """Display tuning results summary.

        Args:
            num_trials (int, optional): Number of trials to display.
                Defaults to 10.
            sort_metric (str, optional): Sorting metric, when not specified
                sort models by objective value. Defaults to None.
        """
        display.section('Results summary')
        display.display_setting('Results in %s' % self._get_project_dir())
        best_trials = self.oracle.get_best_trials(num_trials)
        display.display_setting('Showing %d best trials' % num_trials)
        for trial in best_trials:
            display.display_setting(
                'Objective: {} Score: {}'.format(
                    self.oracle.objective, trial.score))

    @property
    def remaining_trials(self):
        return self.oracle.remaining_trials()

    def get_state(self):
        state = {'stats': self._stats.get_config()}
        return state

    def set_state(self, state):
        self._stats = tuner_utils.TunerStats.from_config(state['stats'])

    def save(self):
        self.oracle.save(self._get_oracle_fname())
        super(Tuner, self).save(self._get_tuner_fname())

    def reload(self):
        self.oracle.reload(self._get_oracle_fname())
        super(Tuner, self).reload(self._get_tuner_fname())

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
            utils.clear_tf_session()
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
            size = utils.maybe_compute_model_size(model)
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

        self.oracle.update_space(hp)
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

    def _checkpoint_trial(self, trial):
        # Write trial status to trial directory
        trial.save(self._get_trial_fname(trial))
        # Send status to Logger
        if self.logger:
            self.logger.report_trial_state(trial.trial_id, trial.get_state())

    def _get_project_dir(self):
        dirname = os.path.join(
            self.directory,
            self.project_name)
        tf_utils.create_directory(dirname)
        return dirname

    def _get_trial_dir(self, trial):
        dirname = os.path.join(
            self._get_project_dir(),
            'trial_' + str(trial.trial_id))
        tf_utils.create_directory(dirname)
        return dirname

    def _get_checkpoint_fname(self, trial, epoch):
        return os.path.join(
            self._get_trial_dir(trial),
            'checkpoints',
            'epoch_' + str(epoch))

    def _get_trial_fname(self, trial):
        return os.path.join(
            self._get_trial_dir(trial),
            'trial.json')

    def _get_tuner_fname(self):
        return os.path.join(
            self._get_project_dir(),
            'tuner_' + str(self.tuner_id) + '.json')

    def _get_oracle_fname(self):
        return os.path.join(
            self._get_project_dir(),
            'oracle_' + str(self.tuner_id) + '.json')

    def _checkpoint_model(self, model, trial, epoch):
        fname = self._get_checkpoint_fname(trial, epoch)
        # Save in TF format.
        model.save_weights(fname)
        return fname
