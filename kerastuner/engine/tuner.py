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

import os
import time
import traceback
import json
import copy

import numpy as np
from tensorflow import keras

from . import hyperparameters as hp_module
from . import hypermodel as hm_module
from . import oracle as oracle_module
from . import trial as trial_module
from . import execution as execution_module
from . import cloudservice as cloudservice_module
from . import tuner_utils
from .. import config
from .. import utils
from ..abstractions import display
from ..abstractions import host as host_module
from ..abstractions.tensorflow import TENSORFLOW_UTILS as tf_utils
from . import metrics_tracking


class Tuner(object):
    """Tuner base class.

    May be subclassed to create new tuners.

    Args:
        oracle: Instance of Oracle class.
        hypermodel: Instance of HyperModel class
            (or callable that takes hyperparameters
            and returns a Model isntance).
        objective: String. Name of model metric to minimize
            or maximize, e.g. "val_accuracy".
        max_trials: Int. Total number of trials
            (model configurations) to test at most.
            Note that the oracle may interrupt the search
            before `max_trial` models have been tested.
        executions_per_trial: Int. Number of executions
            (training a model from scratch,
            starting from a new initialization)
            to run per trial (model configuration).
            Model metrics may vary greatly depending
            on random initialization, hence it is
            often a good idea to run several executions
            per trial in order to evaluate the performance
            of a given set of hyperparameter values.
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
        hyperparameters: HyperParameters class instance.
            Can be used to override (or register in advance)
            hyperparamters in the search space.
        tune_new_entries: Whether hyperparameter entries
            that are requested by the hypermodel
            but that were not specified in `hyperparameters`
            should be added to the search space, or not.
            If not, then the default value for these parameters
            will be used.
        allow_new_entries: Whether the hypermodel is allowed
            to request hyperparameter entries not listed in
            `hyperparameters`.
        distribution_strategy: Optional. A TensorFlow
            `tf.distribute` DistributionStrategy instance. If
            specified, each execution will run under this scope. For
            example, `tf.distribute.MirroredStrategy(['/gpu:0, /'gpu:1])`
            will run each execution on two GPUs. Currently only
            single-worker strategies are supported.
        directory: String. Path to the working directory (relative).
        project_name: Name to use as prefix for files saved
            by this Tuner.
    """

    def __init__(self,
                 oracle,
                 hypermodel,
                 objective,
                 max_trials,
                 executions_per_trial=1,
                 max_model_size=None,
                 optimizer=None,
                 loss=None,
                 metrics=None,
                 hyperparameters=None,
                 tune_new_entries=True,
                 allow_new_entries=True,
                 distribution_strategy=None,
                 directory=None,
                 project_name=None):
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
        self.objective = objective
        self.max_trials = max_trials
        self.executions_per_trial = executions_per_trial
        self.max_model_size = max_model_size
        self.distribution_strategy = distribution_strategy

        # Compilation options
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics

        # Search space management
        if hyperparameters:
            self.hyperparameters = hyperparameters
            self._initial_hyperparameters = hyperparameters.copy()
        else:
            self.hyperparameters = hp_module.HyperParameters()
            self._initial_hyperparameters = None
            if not tune_new_entries:
                raise ValueError(
                    'If you set `tune_new_entries=False`, you must'
                    'specify the search space via the '
                    '`hyperparameters` argument.')
            if not allow_new_entries:
                raise ValueError(
                    'If you set `allow_new_entries=False`, you must'
                    'specify the search space via the '
                    '`hyperparameters` argument.')
        self.tune_new_entries = tune_new_entries
        self.allow_new_entries = allow_new_entries

        # Ops and metadata
        self.directory = directory or '.'
        self.project_name = project_name or 'untitled_project'
        tf_utils.create_directory(
            os.path.join(self.directory, self.project_name))

        # Public internal state
        self.trials = []

        # Private internal state
        self._max_fail_streak = 5
        self._cloudservice = cloudservice_module.CloudService()
        self._stats = tuner_utils.TunerStats()
        self.start_time = int(time.time())

        # Track history of best values per metric
        # (one timestep per trial).
        self.best_metrics = metrics_tracking.MetricsTracker()

        # Logs etc
        self._host = host_module.Host(
            results_dir=os.path.join(self.directory, 'results'),
            tmp_dir=os.path.join(self.directory, 'tmp'),
            export_dir=os.path.join(self.directory, 'export')
        )
        self._display = tuner_utils.Display(self._host)

        # Populate initial search space
        if not self.hyperparameters.space and self.tune_new_entries:
            self._build_model(self.hyperparameters)

    def search(self, *fit_args, **fit_kwargs):
        self.on_search_begin()
        for _ in range(self.max_trials):
            # Obtain unique trial ID to communicate with the oracle.
            trial_id = tuner_utils.generate_trial_id()
            hp = self._call_oracle(trial_id)
            if hp is None:
                # Oracle triggered exit
                return
            trial = trial_module.Trial(
                trial_id=trial_id,
                hyperparameters=hp.copy(),
                max_executions=self.executions_per_trial,
                base_directory=os.path.join(
                    self._host.results_dir, self.project_name)
            )
            self.trials.append(trial)
            self.on_trial_begin(trial)
            self.run_trial(trial, hp, fit_args, fit_kwargs)
            self.on_trial_end(trial)
        self.on_search_end()

    def run_trial(self, trial, hp, fit_args, fit_kwargs):
        fit_kwargs = copy.copy(fit_kwargs)
        original_callbacks = fit_kwargs.get('callbacks', [])[:]
        for i in range(self.executions_per_trial):
            execution_id = tuner_utils.format_execution_id(
                i, self.executions_per_trial)
            # Patch fit arguments
            max_epochs, max_steps = tuner_utils.get_max_epochs_and_steps(
                fit_args, fit_kwargs)
            fit_kwargs['verbose'] = 0

            # Get model; this will reset the Keras session
            if not self.tune_new_entries:
                hp = hp.copy()

            with tuner_utils.maybe_distribute(self.distribution_strategy):
                model = self._build_model(hp)
                self._compile_model(model)

                # Start execution
                execution = execution_module.Execution(
                    execution_id=execution_id,
                    trial_id=trial.trial_id,
                    max_epochs=max_epochs,
                    max_steps=max_steps,
                    base_directory=trial.directory)
                trial.executions.append(execution)
                self.on_execution_begin(trial, execution, model)

                # During model `fit`,
                # the patched callbacks call
                # `self.on_epoch_begin`, `self.on_epoch_end`,
                # `self.on_batch_begin`, `self.on_batch_end`.
                fit_kwargs['callbacks'] = self._inject_callbacks(
                    original_callbacks, trial, execution)
                model.fit(*fit_args, **fit_kwargs)
                self.on_execution_end(trial, execution, model)

            # clean-up TF graph from previously stored (defunct) graph
            utils.clear_tf_session()

    def on_search_begin(self):
        pass

    def on_trial_begin(self, trial):
        pass

    def on_execution_begin(self, trial, execution, model):
        execution.per_epoch_metrics.register_metrics(model.metrics)
        self._display.on_execution_begin(trial, execution, model)

    def on_epoch_begin(self, execution, model, epoch, logs=None):
        # reset per-batch history for the this epoch
        execution.per_batch_metrics = metrics_tracking.MetricsTracker(
            model.metrics)
        self._display.on_epoch_begin(execution, model, epoch, logs=logs)

    def on_batch_begin(self, execution, model, batch, logs):
        pass

    def on_batch_end(self, execution, model, batch, logs=None):
        logs = logs or {}
        for name, value in logs.items():
            execution.per_batch_metrics.update(name, float(value))
        self._checkpoint_execution(execution)
        self._display.on_batch_end(execution, model, batch, logs=logs)

    def on_epoch_end(self, execution, model, epoch, logs=None):
        logs = logs or {}
        # update epoch counters
        execution.epochs_seen += 1

        # update metrics and checkpoint if needed
        for name, value in logs.items():
            improved = execution.per_epoch_metrics.update(name, value)
            if self.objective == name and improved:
                fname = self._checkpoint_model(
                    model, execution.trial_id, execution.execution_id,
                    base_directory=execution.directory)
                execution.best_checkpoint = fname

        # update status
        self._checkpoint_execution(execution, force=True)
        self._display.on_epoch_end(execution, model, epoch, logs=logs)

    def on_execution_end(self, trial, execution, model):
        execution.training_complete = True
        # Update tracker of averaged metrics on Trial
        if len(trial.executions) == 1:
            for name in execution.per_epoch_metrics.names:
                if not trial.averaged_metrics.exists(name):
                    direction = execution.per_epoch_metrics.directions[name]
                    trial.averaged_metrics.register(name, direction)
                trial.averaged_metrics.set_history(
                    name, execution.per_epoch_metrics.get_history(name))
        else:
            # Need to average.
            for name in execution.per_epoch_metrics.names:
                if not trial.averaged_metrics.exists(name):
                    direction = execution.per_epoch_metrics.directions[name]
                    trial.averaged_metrics.register(name, direction)
                histories = []
                for execution in trial.executions:
                    histories.append(
                        execution.per_epoch_metrics.get_history(name))
                # The length of the histories may not match
                # if some executions ran for fewer epochs
                # due to early stopping.
                # We do an average of the available timesteps.
                max_len = max(len(h) for h in histories)
                avg_history = []
                for i in range(max_len):
                    tot = 0.
                    num = 0
                    for h in histories:
                        if len(h) > i:
                            tot += h[i]
                            num += 1
                    avg_history.append(tot / num)
                trial.averaged_metrics.set_history(name, avg_history)

    def on_trial_end(self, trial):
        # Update tracker of best metrics on Tuner
        for name in trial.averaged_metrics.names:
            if not self.best_metrics.exists(name):
                direction = trial.averaged_metrics.directions[name]
                self.best_metrics.register(name, direction)
            self.best_metrics.update(
                name, trial.averaged_metrics.get_best_value(name))
        trial.score = trial.averaged_metrics.get_best_value(self.objective)

        self._checkpoint_trial(trial)
        self._checkpoint_tuner()
        self._display.on_trial_end(
            trial.averaged_metrics,
            self.best_metrics,
            objective=self.objective,
            remaining_trials=self.remaining_trials,
            max_trials=self.max_trials)

    def on_search_end(self):
        pass

    def get_best_models(self, num_models=1):
        """Returns the best model(s), as determined by the tuner's objective.

        The models are loaded with the weights corresponding to
        their best checkpoint (at the end of the best epoch of best execution).

        This method is only a convenience shortcut.

        Args:
            num_models (int, optional): Number of best models to return.
                Models will be returned in sorted order. Defaults to 1.

        Returns:
            List of trained model instances.
        """
        best_trials = self._get_best_trials(num_models)
        models = []
        for trial in best_trials:
            hp = trial.hyperparameters.copy()
            model = self.hypermodel.build(hp)
            self._compile_model(model)
            # Get best execution.
            direction = self.best_metrics.directions[self.objective]
            executions = sorted(
                trial.executions,
                key=lambda x: x.per_epoch_metrics.get_best_value(
                    self.objective),
                reverse=direction == 'max')
            # Reload best checkpoint.
            best_checkpoint = executions[0].best_checkpoint + '-weights.h5'
            model.load_weights(best_checkpoint)
            models.append(model)
        return models

    def search_space_summary(self, extended=False):
        """Print search space summary.

        Args:
            extended: Bool, optional. Display extended summary.
                Defaults to False.
        """
        display.section('Search space summary')
        hp = self.hyperparameters.copy()
        if not hp.space and self.tune_new_entries:
            # Attempt to populate the space
            # if it is expected to be dynamic.
            self.hypermodel.build(hp)
        display.display_setting(
            'Default search space size: %d' % len(hp.space))
        for p in hp.space:
            config = p.get_config()
            name = config.pop('name')
            display.subsection('%s (%s)' % (name, p.__class__.__name__))
            display.display_settings(config)

    def results_summary(self, num_models=10, sort_metric=None):
        """Display tuning results summary.

        Args:
            num_trials (int, optional): Number of trials to display.
                Defaults to 10.
            sort_metric (str, optional): Sorting metric, when not specified
                sort models by objective value. Defaults to None.
        """
        display.section('Results summary')
        if not self.trials:
            display.display_setting('No results to display.')
            return
        display.display_setting('Results in %s' % self._host.results_dir)
        display.display_setting('Ran %d trials' % len(self.trials))
        display.display_setting('Ran %d executions (%d per trial)' %
                                (sum([len(x.executions) for x in self.trials]),
                                 self.executions_per_trial))
        display.display_setting(
            'Best %s: %.4f' % (self.objective,
                               self.best_metrics.get_best_value(
                                   self.objective)))

    def enable_cloud(self, api_key, url=None):
        """Enable cloud service reporting

        Args:
            api_key (str): The backend API access token.
            url (str, optional): The backend base URL.
        """
        self.cloudservice.enable(api_key, url)

    @property
    def remaining_trials(self):
        return self.max_trials - len(self.trials)

    def get_state(self):
        oracle_fname = os.path.join(
            self.directory, self.project_name, 'oracle.json')
        self.oracle.save(oracle_fname)
        oracle_fname = str(oracle_fname)

        state = {
            'oracle': oracle_fname,
            'objective': self.objective,
            'start_time': self.start_time,
            'max_trials': self.max_trials,
            'executions_per_trial': self.executions_per_trial,
            'max_model_size': self.max_model_size,
            # Note that compilation args are not included.
            'hyperparameters': self.hyperparameters.get_config(),
            'tune_new_entries': self.tune_new_entries,
            'allow_new_entries': self.allow_new_entries,
            'directory': str(self.directory),
            'project_name': self.project_name,
            # Dynamic
            'best_metrics': self.best_metrics.get_config(),
            'trials': [t.save() for t in self.trials],
            'start_time': self.start_time,
            # Extra
            'eta': self.eta,
            'remaining_trials': self.remaining_trials,
            'stats': self._stats.get_config(),
            'host': self._host.get_config(),
        }
        best_trials = self._get_best_trials(1)
        if best_trials:
            best_trial = best_trials[0]
            state['best_trial'] = best_trial.get_state()
        else:
            state['best_trial'] = None
        return state

    def save(self):
        fname = os.path.join(self.directory, self.project_name, 'tuner.json')
        state = self.get_state()
        print(state)
        state_json = json.dumps(state)
        tf_utils.write_file(fname, state_json)
        return str(fname)

    def reload(self):
        """Populate `self.trials` and `self.oracle` state."""
        fname = os.path.join(self.directory, self.project_name, 'tuner.json')
        state_data = tf_utils.read_file(fname)
        state = json.loads(state_data)
        self.oracle.reload(state['oracle'])

        self.hyperparameters = hp_module.HyperParameters.from_config(
            state['hyperparameters'])
        self.best_metrics = metrics_tracking.MetricsTracker.from_config(
            state['best_metrics'])
        self.trials = [trial_module.Trial.load(f) for f in state['trials']]
        self.start_time = state['start_time']
        self.stats = tuner_utils.TunerStats.from_config(state['stats'])

    def _call_oracle(self, trial_id):
        if not self.tune_new_entries:
            # In this case, never append to the space
            # so work from a copy of the internal hp object
            hp = self._initial_hyperparameters.copy()
        else:
            # In this case, append to the space,
            # so pass the internal hp object to `build`
            hp = self.hyperparameters

        # Obtain hp value suggestions from the oracle.
        while 1:
            oracle_answer = self.oracle.populate_space(trial_id, hp.space)
            if oracle_answer['status'] == 'RUN':
                hp.values = oracle_answer['values']
                return hp
            elif oracle_answer['status'] == 'EXIT':
                print('Oracle triggered exit')
                return
            else:
                time.sleep(10)

    def _get_best_trials(self, num_trials=1):
        if not self.best_metrics.exists(self.objective):
            return []
        trials = []
        for x in self.trials:
            if x.score is not None:
                trials.append(x)
        if not trials:
            return []
        direction = self.best_metrics.directions[self.objective]
        sorted_trials = sorted(trials,
                               key=lambda x: x.score,
                               reverse=direction == 'max')
        return sorted_trials[:num_trials]

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
            self._stats.num_generated_models += 1
            fail_streak += 1
            try:
                model = self.hypermodel.build(hp)
            except:
                if config.DEBUG:
                    traceback.print_exc()

                self._stats.num_invalid_models += 1
                display.warning('Invalid model %s/%s' %
                                (self._stats.num_invalid_models,
                                 self._max_fail_streak))

                if self._stats.num_invalid_models >= self._max_fail_streak:
                    raise RuntimeError(
                        'Too many failed attempts to build model.')
                continue

            # Check that all mutations of `hp` are legal.
            self._check_space_consistency(hp)

            # Stop if `build()` does not return a valid model.
            if not isinstance(model, keras.models.Model):
                raise RuntimeError(
                    'Model-building function did not return '
                    'a valid Model instance.')

            # Check model size.
            size = utils.compute_model_size(model)
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

    def _check_space_consistency(self, hp):
        # Optionally disallow hyperparameters defined on the fly.
        if not self._initial_hyperparameters:
            return
        old_space = [x.name for x in self._initial_hyperparameters.space]
        new_space = [x.name for x in hp.space]
        difference = set(new_space) - set(old_space)
        if not self.allow_new_entries and difference:
            diff = set(new_space) - set(old_space)
            raise RuntimeError(
                'The hypermodel has requested a parameter that was not part '
                'of `hyperparameters`, '
                'yet `allow_new_entries` is set to False. '
                'The unknown parameters are: {diff}'.format(diff=diff))
        # TODO: consider calling the oracle to update the space.

    @property
    def eta(self):
        """Return search estimated completion time."""
        num_trials = len(self.trials)
        num_remaining_trials = self.max_trials - num_trials
        if num_remaining_trials < 1:
            return 0
        else:
            elapsed_time = int(time.time() - self.start_time)
            time_per_trial = elapsed_time / num_trials
            return int(num_remaining_trials * time_per_trial)

    def _inject_callbacks(self, callbacks, trial, execution):
        # Deepcopy and patch callbacks if needed
        if callbacks:
            try:
                callbacks = copy.deepcopy(callbacks)
            except:
                raise ValueError(
                    'All callbacks used during a search '
                    'should be deep-copyable (since they are '
                    'reused across executions). '
                    'It is not possible to do `copy.deepcopy(%s)`' %
                    (callbacks,))
            for callback in callbacks:
                # patching tensorboard log dir
                if callback.__class__.__name__ == 'TensorBoard':
                    tensorboard_idx = '%s-%s' % (trial.trial_id,
                                                 execution.execution_id)
                    callback.log_dir = os.path.join(execution.directory,
                                                    tensorboard_idx)
        else:
            callbacks = []

        # Add callback to call back the tuner during `fit`.
        callbacks.append(tuner_utils.TunerCallback(self, trial, execution))
        return callbacks

    def _checkpoint_tuner(self):
        # Write tuner status to tuner directory
        self.save()
        # Send status to cloudservice
        if self._cloudservice and self._cloudservice.enabled:
            # TODO
            state = self.get_state()
            self._cloudservice.send_tuner_status(status)

    def _checkpoint_trial(self, trial):
        # Write trial status to trial directory
        trial.save()
        # Send status to cloudservice
        if self._cloudservice and self._cloudservice.enabled:
            # TODO
            state = trial.get_state()
            self._cloudservice.send_trial_status(status)

    def _checkpoint_execution(self, execution, force=False):
        refresh_interval = 30
        if hasattr(self, '_last_execution_status_refresh'):
            elapsed = int(time.time()) - self._last_execution_status_refresh
            if refresh_interval > elapsed and not force:
                return
            self._last_execution_status_refresh = int(time.time())

        # Write execution status to execution directory
        execution.save()
        # Send status to cloudservice
        if self._cloudservice and self._cloudservice.enabled:
            # TODO
            state = execution.get_state()
            self._cloudservice.send_execution_status(state)

    def _checkpoint_model(self, model, trial_id, execution_id,
                          base_directory='.'):
        file_prefix = '%s-%s' % (trial_id, execution_id)
        base_filename = os.path.join(base_directory, file_prefix)

        tmp_path = os.path.join(self._host.tmp_dir,
                                file_prefix)

        try:
            tf_utils.save_model(model,
                                base_filename,
                                tmp_path=tmp_path,
                                export_type='keras')
        except:
            traceback.print_exc()
            display.write_log('Checkpoint failed.')
            exit(0)
        return base_filename
