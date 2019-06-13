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

from . import hyperparameters as hp_module
from . import hypermodel as hm_module
from . import trial as trial_module
from . import execution as execution_module
from . import config as engine_config
from .. import cloudservice as cloudservice_module


class Tuner(object):
    """Tuner base class.
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
                 allow_new_parameters=True,
                 directory=None,
                 project_name=None,
                 metadata=None):
        self.oracle = oracle  # TODO: check Oracle instance
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
        self.executions_per_trial = 1
        self.max_model_size = max_model_size

        # Compilation options
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics

        # Search space management
        if hyperparameters:
            self.hyperparameters = hyperparameters
            self._initial_hyperparameters = hp_module.HyperParameters.from_config(
                hyperparameters.get_config())
        else:
            self.hyperparameters = hp_module.HyperParameters()
            self._initial_hyperparameters = None
        self.allow_new_parameters = allow_new_parameters

        # Ops and metadata
        self.directory = directory
        self.project_name = project_name
        self.metadata = metadata

        # Internal state
        self._trials = []
        self._max_fail_streak = 5
        self._cloudservice = None  # TODO / cloudservice_module.CloudService()
        self._stats = TunerStats()
        self._host = None  # TODO
        self._aggregated_metrics = None
        self._best_trial = None
        self._start_time = int(time())
        self._num_executions = 0

        # Log file
        log_name = '%s-%d.log' % (self.project_name, self.start_time)
        self._log_file = os.path.join(self._host.results_dir, log_name)
        set_log(self._log_file)

    def search(self, *fit_args, **fit_kwargs):
        """Top-level loop of the search process.

        Can be overridden by subclass implementers.

        Args:
            *fit_args: To be passed to `fit`.
            **fit_kwargs: To be passed to `fit`.

        Note that any callbacks will be pickled so as to be reused
        across executions.
        """
        if self._initial_hyperparameters:
            # In this case, never append to the space
            # so work from a copy of the internal hp object
            hp = hp_module.HyperParameters.from_config(
                self._initial_hyperparameters.get_config())
        else:
            # In this case, append to the space,
            # so pass the internal hp object to `build`
            hp = self.hyperparameters
        for i in range(self.max_trials):
            trial_id = None  # TODO
            hp.values = self.oracle.populate_space(hp.space, trial_id)
            # These 2 lines can be turned into async calls
            # in the distributed case: we dispatch trials,
            # and get back the result at some later point
            self._run_trial(hp, *fit_args, **fit_kwargs)
        if self.cloudservice.enabled:
            self.cloudservice.complete()

    def run_trial(self, hp, trial_id, *fit_args, **fit_kwargs):
        """Runs one trial (may cover multiple Executions of the same model).

        Can be overridden by subclass implementers.

        Args:
            hp: Populated HyperParameters objects holding the
                parameter values to be used for this trial.
            *fit_args: To be passed to `fit`.
            **fit_kwargs: To be passed to `fit`.
        """
        model = self._build_model(hp)
        trial = self._add_trial(model, hp)
        trial.run_execution(*fit_args, **fit_kwargs)
        score = trial.best_metrics[self.objective]
        self.oracle.result(trial_id, score)

    def _add_trial(self, model, hp):
        # TODO
        pass

    def get_best_models(self, num_models=1, compile=True):
        """Returns the best model(s), as determined by the tuner's objective.

        The models are loaded with the weights corresponding to
        their best checkpoint.

        This method is only a convenience shortcut.

        Args:
            num_models (int, optional): Number of best models to return.
                Models will be returned in sorted order. Defaults to 1.
            compile (bool, optional): If True, compile the returned models
                using the loss, optimizer, and metrics used during training.
                Defaults to True.

        Returns:
            List of trained model instances.
        """
        # TODO
        pass

    def search_space_summary(self, extended=False):
        """Print tuner summary

        Args:
            extended: Bool, optional. Display extended summary.
                Defaults to False.
        """
        # TODO
        pass

    def results_summary(self, num_models=10, sort_metric=None):
        """Display tuning results summary.

        Args:
            num_trials (int, optional): Number of trials to display.
                Defaults to 10.
            sort_metric (str, optional): Sorting metric, when not specified
                sort models by objective value. Defaults to None.
        """
        # TODO
        pass

    def enable_cloud(self, api_key, url=None):
        """Enable cloud service reporting

        Args:
            api_key (str): The backend API access token.
            url (str, optional): The backend base URL.
        """
        self.cloudservice.enable(api_key, url)

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
        collision_streak = 0
        oversized_streak = 0

        while 1:
            # clean-up TF graph from previously stored (defunct) graph
            tf_utils.clear_tf_session()
            self._stats.generated_instances += 1
            fail_streak += 1
            try:
                model = self.hypermodel.build(hp)
            except:
                if engine_config.DEBUG:
                    traceback.print_exc()

                self._stats.invalid_instances += 1
                warning('Invalid model %s/%s' %
                        (self._stats.invalid_instances, self._max_fail_streak))

                if self._stats.invalid_instances >= self._max_fail_streak:
                    raise RuntimeError(
                        'Too many failed attempts to build model.')
                continue

            # Check that all mutations of `hp` are legal.
            self._check_space_consistency(hp)

            # Stop if `build()` does not return a valid model.
            if not isinstance(model, Model):
                raise RuntimeError(
                    'Model-building function did not return '
                    'a valid Model instance.')

            # Get unique id for instance.
            idx = self._compute_model_id(model)
            if idx in self._model_ids:
                collision_streak += 1
                self._stats.collisions += 1
                warning('Collision for %s -- skipping' % (idx))
                if collision_streak >= self._max_fail_streak:
                    raise RuntimeError(
                        'Too many identical models generated.')
                continue

            # Check model size.
            size = tf_utils.compute_model_size(model)
            if size > self.max_model_size:
                oversized_streak += 1
                self._stats.oversized_models += 1
                warning('Oversized model: %s parameters -- skipping' % (size))
                if oversized_streak >= self._max_fail_streak:
                    raise RuntimeError('Too many consecutive oversize models.')
                continue

        return self._compile_model(model)

    def _compile_model(model):
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
        old_space = self._initial_hyperparameters.space[:]
        new_space = hyperparameters.space[:]
        difference = set(new_space) - set(old_space)
        if not self.allow_new_parameters and difference:
            diff = set(new_space) - set(old_space)
            raise RuntimeError(
                'The hypermodel has requested a parameter that was not part '
                'of `hyperparameters`, '
                'yet `allow_new_parameters` is set to False. '
                'The unknown parameters are: {diff}'.format(diff=diff))
        self.oracle.update_space(difference)

    def _compute_model_id(self, model):
        """Compute unique model hash."""
        s = str(model.get_config())
        # Optimizer and loss are not currently part of the model config,
        # but could conceivably be part of the model_fn/tuning process.
        if model.optimizer:
            s += 'optimizer:' + str(model.optimizer.get_config())
        s += 'loss:' + str(model.loss)
        return hashlib.sha256(s.encode('utf-8')).hexdigest()[:32]

    @property
    def _eta(self):
        """Return search estimated completion time.
        """
        num_trials = len(self._trials)
        num_remaining_trials = self.max_trials - num_trials
        if num_remaining_trials < 1:
            return 0
        else:
            elapsed_time = int(time() - self._start_time)
            time_per_trial = elapsed_time / num_trials
            return int(num_remaining_trials * time_per_epoch)


class TunerStats(object):
    """Track tuner statistics."""

    def __init__(self):
        self.generated_instances = 0  # overall number of instances generated
        self.invalid_instances = 0  # how many models didn't work
        self.instances_previously_trained = 0  # num instance already trained
        self.collisions = 0  # how many time we regenerated the same model
        self.oversized_models = 0  # num models with params> max_params

    def summary(self, extended=False):
        subsection('Tuning stats')
        display_settings(self.get_config())

    def get_config(self):
        return {
            'num_generated_models': self.generated_instances,
            'num_invalid_models': self.invalid_instances,
            'num_mdl_previously_trained': self.instances_previously_trained,
            'num_collision': self.collisions,
            'num_over_sized_models': self.over_sized_models
        }
