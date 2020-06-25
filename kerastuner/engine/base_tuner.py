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
import tensorflow as tf

from .. import utils
from ..distribute import utils as dist_utils
from ..distribute import oracle_chief
from ..distribute import oracle_client
from . import hypermodel as hm_module
from . import oracle as oracle_module
from . import stateful
from . import trial as trial_module
from . import tuner_utils


class BaseTuner(stateful.Stateful):
    """Tuner base class.

    May be subclassed to create new tuners, including for non-Keras models.

    # Arguments:
        oracle: Instance of Oracle class.
        hypermodel: Instance of HyperModel class
            (or callable that takes hyperparameters
            and returns a Model instance).
        directory: String. Path to the working directory (relative).
        project_name: Name to use as prefix for files saved
            by this Tuner.
        logger: Optional. Instance of Logger class, used for streaming data
            to Cloud Service for monitoring.
        overwrite: Bool, default `False`. If `False`, reloads an existing project
            of the same name if one is found. Otherwise, overwrites the project.
    """

    def __init__(self,
                 oracle,
                 hypermodel,
                 directory=None,
                 project_name=None,
                 logger=None,
                 overwrite=False):
        # Ops and metadata
        self.directory = directory or '.'
        self.project_name = project_name or 'untitled_project'
        if overwrite and tf.io.gfile.exists(self.project_dir):
            tf.io.gfile.rmtree(self.project_dir)

        if not isinstance(oracle, oracle_module.Oracle):
            raise ValueError('Expected oracle to be '
                             'an instance of Oracle, got: %s' % (oracle,))
        self.oracle = oracle
        self.oracle._set_project_dir(
            self.directory, self.project_name, overwrite=overwrite)

        # Run in distributed mode.
        if dist_utils.is_chief_oracle():
            # Blocks forever.
            oracle_chief.start_server(self.oracle)
        elif dist_utils.has_chief_oracle():
            # Proxies requests to the chief oracle.
            self.oracle = oracle_client.OracleClient(self.oracle)

        # To support tuning distribution.
        self.tuner_id = os.environ.get('KERASTUNER_TUNER_ID', 'tuner0')

        self.hypermodel = hm_module.get_hypermodel(hypermodel)

        # Logs etc
        self.logger = logger
        self._display = tuner_utils.Display(oracle=self.oracle)

        self._populate_initial_space()

        if not overwrite and tf.io.gfile.exists(self._get_tuner_fname()):
            tf.get_logger().info('Reloading Tuner from {}'.format(
                self._get_tuner_fname()))
            self.reload()

    def _populate_initial_space(self):
        """Populate initial search space for oracle.

        Keep this function as a subroutine for AutoKeras to override. The space may
        not be ready at the initialization of the tuner, but after seeing the
        training data.
        """
        hp = self.oracle.get_space()
        self.hypermodel.build(hp)
        self.oracle.update_space(hp)

    def search(self, *fit_args, **fit_kwargs):
        """Performs a search for best hyperparameter configuations.

        # Arguments:
            *fit_args: Positional arguments that should be passed to
              `run_trial`, for example the training and validation data.
            *fit_kwargs: Keyword arguments that should be passed to
              `run_trial`, for example the training and validation data.
        """
        if 'verbose' in fit_kwargs:
            self._display.verbose = fit_kwargs.get('verbose')
        self.on_search_begin()
        while True:
            trial = self.oracle.create_trial(self.tuner_id)
            if trial.status == trial_module.TrialStatus.STOPPED:
                # Oracle triggered exit.
                tf.get_logger().info('Oracle triggered exit')
                break
            if trial.status == trial_module.TrialStatus.IDLE:
                # Oracle is calculating, resend request.
                continue

            self.on_trial_begin(trial)
            self.run_trial(trial, *fit_args, **fit_kwargs)
            self.on_trial_end(trial)
        self.on_search_end()

    def run_trial(self, trial, *fit_args, **fit_kwargs):
        """Evaluates a set of hyperparameter values.

        This method is called during `search` to evaluate a set of
        hyperparameters.

        For subclass implementers: This method is responsible for
        reporting metrics related to the `Trial` to the `Oracle`
        via `self.oracle.update_trial`.

        Simplest example:

        ```python
        def run_trial(self, trial, x, y, val_x, val_y):
            model = self.hypermodel.build(trial.hyperparameters)
            model.fit(x, y)
            loss = model.evaluate(val_x, val_y)
            self.oracle.update_trial(
              trial.trial_id, {'loss': loss})
            self.save_model(trial.trial_id, model)
        ```

        # Arguments:
            trial: A `Trial` instance that contains the information
              needed to run this trial. Hyperparameters can be accessed
              via `trial.hyperparameters`.
            *fit_args: Positional arguments passed by `search`.
            *fit_kwargs: Keyword arguments passed by `search`.
        """
        raise NotImplementedError

    def save_model(self, trial_id, model, step=0):
        """Saves a Model for a given trial.

        # Arguments:
            trial_id: The ID of the `Trial` that corresponds to this Model.
            model: The trained model.
            step: For models that report intermediate results to the `Oracle`,
              the step that this saved file should correspond to. For example,
              for Keras models this is the number of epochs trained.
        """
        raise NotImplementedError

    def load_model(self, trial):
        """Loads a Model from a given trial.

        # Arguments:
            trial: A `Trial` instance. For models that report intermediate
              results to the `Oracle`, generally `load_model` should load the
              best reported `step` by relying of `trial.best_step`
        """
        raise NotImplementedError

    def on_search_begin(self):
        """A hook called at the beginning of `search`."""
        if self.logger:
            self.logger.register_tuner(self.get_state())

    def on_trial_begin(self, trial):
        """A hook called before starting each trial.

        # Arguments:
            trial: A `Trial` instance.
        """
        if self.logger:
            self.logger.register_trial(trial.trial_id, trial.get_state())
        self._display.on_trial_begin(self.oracle.get_trial(trial.trial_id))

    def on_trial_end(self, trial):
        """A hook called after each trial is run.

        # Arguments:
            trial: A `Trial` instance.
        """
        # Send status to Logger
        if self.logger:
            self.logger.report_trial_state(trial.trial_id, trial.get_state())

        self.oracle.end_trial(
            trial.trial_id, trial_module.TrialStatus.COMPLETED)
        self.oracle.update_space(trial.hyperparameters)
        # Display needs the updated trial scored by the Oracle.
        self._display.on_trial_end(self.oracle.get_trial(trial.trial_id))
        self.save()

    def on_search_end(self):
        """A hook called at the end of `search`."""
        if self.logger:
            self.logger.exit()

    def get_best_models(self, num_models=1):
        """Returns the best model(s), as determined by the objective.

        This method is only a convenience shortcut. For best performance, It is
        recommended to retrain your Model on the full dataset using the best
        hyperparameters found during `search`.

        # Arguments:
            num_models (int, optional). Number of best models to return.
                Models will be returned in sorted order. Defaults to 1.

        # Returns:
            List of trained model instances.
        """
        best_trials = self.oracle.get_best_trials(num_models)
        models = [self.load_model(trial) for trial in best_trials]
        return models

    def get_best_hyperparameters(self, num_trials=1):
        """Returns the best hyperparameters, as determined by the objective.

        This method can be used to reinstantiate the (untrained) best model
        found during the search process.

        Example:

        ```python
        best_hp = tuner.get_best_hyperparameters()[0]
        model = tuner.hypermodel.build(best_hp)
        ```

        # Arguments:
            num_trials: (int, optional). Number of `HyperParameters` objects to
              return. `HyperParameters` will be returned in sorted order based on
              trial performance.

        # Returns:
            List of `HyperParameter` objects.
        """
        return [t.hyperparameters for t in self.oracle.get_best_trials(num_trials)]

    def search_space_summary(self, extended=False):
        """Print search space summary.

        Args:
            extended: Bool, optional. Display extended summary.
                Defaults to False.
        """
        print('Search space summary')
        hp = self.oracle.get_space()
        print('Default search space size: %d' % len(hp.space))
        for p in hp.space:
            config = p.get_config()
            name = config.pop('name')
            print('%s (%s)' % (name, p.__class__.__name__))
            print(config)

    def results_summary(self, num_trials=10):
        """Display tuning results summary.

        Args:
            num_trials (int, optional): Number of trials to display.
                Defaults to 10.
        """
        print('Results summary')
        print('Results in %s' % self.project_dir)
        print('Showing %d best trials' % num_trials)
        print('{}'.format(self.oracle.objective))

        best_trials = self.oracle.get_best_trials(num_trials)
        for trial in best_trials:
            trial.summary()

    @property
    def remaining_trials(self):
        """Returns the number of trials remaining.

        Will return `None` if `max_trials` is not set.
        """
        return self.oracle.remaining_trials()

    def get_state(self):
        return {}

    def set_state(self, state):
        pass

    def save(self):
        """Saves this object to its project directory."""
        if not dist_utils.has_chief_oracle():
            self.oracle.save()
        super(BaseTuner, self).save(self._get_tuner_fname())

    def reload(self):
        """Reloads this object from its project directory."""
        if not dist_utils.has_chief_oracle():
            self.oracle.reload()
        super(BaseTuner, self).reload(self._get_tuner_fname())

    @property
    def project_dir(self):
        dirname = os.path.join(
            str(self.directory),
            self.project_name)
        utils.create_directory(dirname)
        return dirname

    def get_trial_dir(self, trial_id):
        dirname = os.path.join(
            str(self.project_dir),
            'trial_' + str(trial_id))
        utils.create_directory(dirname)
        return dirname

    def _get_tuner_fname(self):
        return os.path.join(
            str(self.project_dir),
            str(self.tuner_id) + '.json')
