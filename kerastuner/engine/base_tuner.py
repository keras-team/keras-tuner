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

from .. import utils
from ..abstractions import display
from . import hypermodel as hm_module
from . import oracle as oracle_module
from . import stateful
from . import trial as trial_module
from . import tuner_utils


class BaseTuner(stateful.Stateful):
    """Tuner base class.

    May be subclassed to create new tuners, including for non-Keras models.

    Args:
        oracle: Instance of Oracle class.
        hypermodel: Instance of HyperModel class
            (or callable that takes hyperparameters
            and returns a Model instance).
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

        # To support tuning distribution.
        self.tuner_id = tuner_id if tuner_id is not None else 0

        # Ops and metadata
        self.directory = directory or '.'
        self.project_name = project_name or 'untitled_project'
        self.oracle.set_project_dir(self.directory, self.project_name)

        # Logs etc
        self.logger = logger
        self._display = tuner_utils.Display()

        # Populate initial search space.
        hp = self.oracle.get_space()
        self.hypermodel.build(hp)
        self.oracle.update_space(hp)

    def search(self, *fit_args, **fit_kwargs):
        self.on_search_begin()
        while True:
            trial = self.oracle.create_trial(self.tuner_id)
            if trial.status == trial_module.TrialStatus.STOPPED:
                # Oracle triggered exit.
                print('Oracle triggerd exit')
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
        """
        raise NotImplementedError

    def save_model(self, trial_id, model, step=0):
        raise NotImplementedError

    def load_model(self, trial):
        raise NotImplementedError

    def on_search_begin(self):
        if self.logger:
            self.logger.register_tuner(self.get_state())

    def on_trial_begin(self, trial):
        if self.logger:
            self.logger.register_trial(trial.trial_id, trial.get_state())

    def on_trial_end(self, trial):
        # Send status to Logger
        if self.logger:
            self.logger.report_trial_state(trial.trial_id, trial.get_state())

        self.oracle.end_trial(
            trial.trial_id, trial_module.TrialStatus.COMPLETED)
        self.oracle.update_space(trial.hyperparameters)
        self._display.on_trial_end(trial)
        self.save()

    def on_search_end(self):
        if self.logger:
            self.logger.exit()

    def get_best_models(self, num_models=1):
        """Returns the best model(s), as determined by the tuner's objective.

        This method is only a convenience shortcut.

        Args:
            num_models (int, optional): Number of best models to return.
                Models will be returned in sorted order. Defaults to 1.

        Returns:
            List of trained model instances.
        """
        best_trials = self.oracle.get_best_trials(num_models)
        models = [self.load_model(trial) for trial in best_trials]
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
        display.display_setting('Results in %s' % self.project_dir)
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
        return {}

    def set_state(self, state):
        pass

    def save(self):
        super(BaseTuner, self).save(self._get_tuner_fname())

    def reload(self):
        super(BaseTuner, self).reload(self._get_tuner_fname())

    @property
    def project_dir(self):
        dirname = os.path.join(
            self.directory,
            self.project_name)
        utils.create_directory(dirname)
        return dirname

    def get_trial_dir(self, trial_id):
        dirname = os.path.join(
            self.project_dir,
            'trial_' + str(trial_id))
        utils.create_directory(dirname)
        return dirname

    def _get_tuner_fname(self):
        return os.path.join(
            self.project_dir,
            'tuner_' + str(self.tuner_id) + '.json')
