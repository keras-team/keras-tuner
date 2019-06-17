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
import random
import hashlib
import traceback

from tensorflow import keras

from . import hyperparameters as hp_module
from . import hypermodel as hm_module
from . import oracle as oracle_module
from . import trial as trial_module
from . import execution as execution_module
from . import cloudservice as cloudservice_module
from .. import config
from .. import utils
from ..abstractions import display
from ..abstractions import host as host_module
from . import metrics_tracking


class Tuner(object):
    """Tuner base class.

    May be subclassed to create new tuners.

    Args:
        oracle: Instance of Oracle class.
        hypermodel: Instnace of HyperModel class
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

        # Public internal state
        self.trials = []

        # Private internal state
        self._max_fail_streak = 5
        self._cloudservice = cloudservice_module.CloudService()
        self._stats = TunerStats()
        self._best_metrics = None
        self._best_trial = None
        self._start_time = int(time.time())
        self._num_executions = 0

        # Track history of best values per metric
        # (one timestep per trial).
        self.best_metrics = metrics_tracking.MetricsTracker()

        # Logs etc
        self._host = host_module.Host(
            results_dir=os.path.join(self.directory, 'results'),
            tmp_dir=os.path.join(self.directory, 'tmp'),
            export_dir=os.path.join(self.directory, 'export')
        )
        log_name = '%s-%d.log' % (self.project_name, self._start_time)
        self._log_file = os.path.join(self._host.results_dir, log_name)

    def search(self, *fit_args, **fit_kwargs):
        """Top-level loop of the search process.

        Can be overridden by subclass implementers.

        Args:
            *fit_args: To be passed to `fit`.
            **fit_kwargs: To be passed to `fit`.

        Note that any callbacks will be pickled so as to be reused
        across executions.
        """
        if not self.tune_new_entries:
            # In this case, never append to the space
            # so work from a copy of the internal hp object
            hp = self._initial_hyperparameters.copy()
        else:
            # In this case, append to the space,
            # so pass the internal hp object to `build`
            hp = self.hyperparameters
        for i in range(self.max_trials):
            # Obtain unique trial ID to communicate with the oracle.
            trial_id = self._generate_trial_id()
            # Obtain hp value suggestions from the oracle.
            while 1:
                oracle_answer = self.oracle.populate_space(trial_id, hp.space)
                if oracle_answer['status'] == 'RUN':
                    hp.values = oracle_answer['values']
                    break
                elif oracle_answer['status'] == 'EXIT':
                    print('Oracle triggered exit')
                    return
                else:
                    time.sleep(10)
            # Run trial with hp value suggestions.
            trial = self.run_trial(trial_id, hp, *fit_args, **fit_kwargs)
            score = trial.averaged_metrics.get_best_value(self.objective)
            self.oracle.result(trial_id, score)

        if self._cloudservice.enabled:
            self._cloudservice.complete()

    def run_trial(self, trial_id, hp, *fit_args, **fit_kwargs):
        """Runs one trial (may cover multiple Executions of the same model).

        Can be overridden by subclass implementers.

        Args:
            trial_id: String. Unique trial id.
            hp: Populated HyperParameters objects holding the
                parameter values to be used for this trial.
            *fit_args: To be passed to `fit`.
            **fit_kwargs: To be passed to `fit`.

        Returns:
            Trial instance.
        """
        # Create a sample model from the hp configuration.
        # Note that this may add new entries to the search space.
        model = self._build_model(hp)
        trial = trial_module.Trial(trial_id, hp, model, self.objective,
                                   tuner=self, cloudservice=self._cloudservice)
        self.trials.append(trial)
        for _ in range(self.executions_per_trial):
            execution = trial.run_execution(*fit_args, **fit_kwargs)
        return trial

    def get_best_models(self, num_models=1):
        """Returns the best model(s), as determined by the tuner's objective.

        The models are loaded with the weights corresponding to
        their best checkpoint.

        This method is only a convenience shortcut.

        Args:
            num_models (int, optional): Number of best models to return.
                Models will be returned in sorted order. Defaults to 1.

        Returns:
            List of trained model instances.
        """
        best_trials = self._get_best_trials(num_models)
        hps = [x.hyperparameters for x in best_trials]
        models = [self.hypermodel.build(hp) for hp in hps]
        for model in models:
            self._compile_model(model)
        # TODO: reload best checkpoint.
        return models

    def search_space_summary(self, extended=False):
        """Print search space summary.

        Args:
            extended: Bool, optional. Display extended summary.
                Defaults to False.
        """
        display.section('Search space summary')
        hp = self.hyperparameters.copy()
        if self.allow_new_entries:
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

    def get_status(self):
        config = {
            'project_name': self.project_name,
            'objective': self.objective,
            'start_time': self._start_time,
            'max_trials': self.max_trials,
            'executions_per_trial': self.executions_per_trial,
            'eta': self._eta,
            'remaining_trials': self.remaining_trials,
        }

        config['stats'] = self._stats.get_config()
        config['host'] = self._host.get_config()
        config['metrics'] = {}

        best_trials = self._get_best_trials(1)
        if best_trials:
            config['aggregate_metrics'] = self.best_metrics.get_config()
            best_trial = best_trials[0]
            config['best_trial'] = best_trial.get_status()
        else:
            config['aggregate_metrics'] = []
            config['best_trial'] = None
        return config

    @property
    def remaining_trials(self):
        return self.max_trials - len(self.trials)

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
        return trials[:num_trials]

    def _generate_trial_id(self):
        s = str(time.time()) + str(random.randint(1, 1e7))
        return hashlib.sha256(s.encode('utf-8')).hexdigest()[:32]

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
    def _eta(self):
        """Return search estimated completion time.
        """
        num_trials = len(self.trials)
        num_remaining_trials = self.max_trials - num_trials
        if num_remaining_trials < 1:
            return 0
        else:
            elapsed_time = int(time.time() - self._start_time)
            time_per_trial = elapsed_time / num_trials
            return int(num_remaining_trials * time_per_trial)


class TunerStats(object):
    """Track tuner statistics."""

    def __init__(self):
        self.num_generated_models = 0  # overall number of instances generated
        self.num_invalid_models = 0  # how many models didn't work
        self.num_oversized_models = 0  # num models with params> max_params

    def summary(self, extended=False):
        subsection('Tuning stats')
        display_settings(self.get_config())

    def get_config(self):
        return {
            'num_generated_models': self.num_generated_models,
            'num_invalid_models': self.num_invalid_models,
            'num_oversized_models': self.num_oversized_models
        }
