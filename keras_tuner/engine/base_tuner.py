# Copyright 2019 The KerasTuner Authors
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


import copy
import os
import warnings

import tensorflow as tf

from keras_tuner import utils
from keras_tuner.distribute import oracle_chief
from keras_tuner.distribute import oracle_client
from keras_tuner.distribute import utils as dist_utils
from keras_tuner.engine import hypermodel as hm_module
from keras_tuner.engine import oracle as oracle_module
from keras_tuner.engine import stateful
from keras_tuner.engine import trial as trial_module
from keras_tuner.engine import tuner_utils


class BaseTuner(stateful.Stateful):
    """Tuner base class.

    `BaseTuner` is the base class for all Tuners, which manages the search
    loop, Oracle, logging, saving, etc. Tuners for non-Keras models can be
    created by subclassing `BaseTuner`.

    Args:
        oracle: Instance of Oracle class.
        hypermodel: Instance of `HyperModel` class (or callable that takes
            hyperparameters and returns a `Model` instance). It is optional
            when `Tuner.run_trial()` is overriden and does not use
            `self.hypermodel`.
        directory: A string, the relative path to the working directory.
        project_name: A string, the name to use as prefix for files saved by
            this Tuner.
        logger: Optional instance of `kerastuner.Logger` class for
            streaming logs for monitoring.
        overwrite: Boolean, defaults to `False`. If `False`, reloads an
            existing project of the same name if one is found. Otherwise,
            overwrites the project.

    Attributes:
        remaining_trials: Number of trials remaining, `None` if `max_trials` is
            not set. This is useful when resuming a previously stopped search.
    """

    def __init__(
        self,
        oracle,
        hypermodel=None,
        directory=None,
        project_name=None,
        logger=None,
        overwrite=False,
    ):
        # Ops and metadata
        self.directory = directory or "."
        self.project_name = project_name or "untitled_project"
        if overwrite and tf.io.gfile.exists(self.project_dir):
            tf.io.gfile.rmtree(self.project_dir)

        if not isinstance(oracle, oracle_module.Oracle):
            raise ValueError(
                "Expected `oracle` argument to be an instance of `Oracle`. "
                f"Received: oracle={oracle} (of type ({type(oracle)})."
            )
        self.oracle = oracle
        self.oracle._set_project_dir(
            self.directory, self.project_name, overwrite=overwrite
        )

        # Run in distributed mode.
        if dist_utils.is_chief_oracle():
            # Blocks forever.
            oracle_chief.start_server(self.oracle)
        elif dist_utils.has_chief_oracle():
            # Proxies requests to the chief oracle.
            self.oracle = oracle_client.OracleClient(self.oracle)

        # To support tuning distribution.
        self.tuner_id = os.environ.get("KERASTUNER_TUNER_ID", "tuner0")

        self.hypermodel = hm_module.get_hypermodel(hypermodel)

        # Logs etc
        self.logger = logger
        self._display = tuner_utils.Display(oracle=self.oracle)

        self._populate_initial_space()

        if not overwrite and tf.io.gfile.exists(self._get_tuner_fname()):
            tf.get_logger().info(
                "Reloading Tuner from {}".format(self._get_tuner_fname())
            )
            self.reload()

    def _populate_initial_space(self):
        """Populate initial search space for oracle.

        Keep this function as a subroutine for AutoKeras to override. The space
        may not be ready at the initialization of the tuner, but after seeing
        the training data.

        Build hypermodel multiple times to find all conditional hps. It
        generates hp values based on the not activated `conditional_scopes`
        found in the builds.
        """
        if self.hypermodel is None:
            return

        hp = self.oracle.get_space()

        # Lists of stacks of conditions used during `explore_space()`.
        scopes_never_active = []
        scopes_once_active = []

        while True:
            self.hypermodel.build(hp)

            # Update the recored scopes.
            for conditions in hp.active_scopes:
                if conditions not in scopes_once_active:
                    scopes_once_active.append(copy.deepcopy(conditions))
                if conditions in scopes_never_active:
                    scopes_never_active.remove(conditions)

            for conditions in hp.inactive_scopes:
                if conditions not in scopes_once_active:
                    scopes_never_active.append(copy.deepcopy(conditions))

            # All conditional scopes are activated.
            if len(scopes_never_active) == 0:
                break

            # Generate new values to activate new conditions.
            conditions = scopes_never_active[0]
            for condition in conditions:
                hp.values[condition.name] = condition.values[0]

        self.oracle.update_space(hp)

    def search(self, *fit_args, **fit_kwargs):
        """Performs a search for best hyperparameter configuations.

        Args:
            *fit_args: Positional arguments that should be passed to
              `run_trial`, for example the training and validation data.
            **fit_kwargs: Keyword arguments that should be passed to
              `run_trial`, for example the training and validation data.
        """
        if "verbose" in fit_kwargs:
            self._display.verbose = fit_kwargs.get("verbose")
        self.on_search_begin()
        while True:
            trial = self.oracle.create_trial(self.tuner_id)
            if trial.status == trial_module.TrialStatus.STOPPED:
                # Oracle triggered exit.
                tf.get_logger().info("Oracle triggered exit")
                break
            if trial.status == trial_module.TrialStatus.IDLE:
                # Oracle is calculating, resend request.
                continue

            self.on_trial_begin(trial)
            results = self.run_trial(trial, *fit_args, **fit_kwargs)
            # `results` is None indicates user updated oracle in `run_trial()`.
            if results is None:
                warnings.warn(
                    "`Tuner.run_trial()` returned None. It should return one of "
                    "float, dict, keras.callbacks.History, or a list of one "
                    "of these types. The use case of calling "
                    "`Tuner.oracle.update_trial()` in `Tuner.run_trial()` is "
                    "deprecated, and will be removed in the future.",
                    DeprecationWarning,
                    stacklevel=2,
                )
            else:
                self.oracle.update_trial(
                    trial.trial_id,
                    # Convert to dictionary before calling `update_trial()`
                    # to pass it from gRPC.
                    tuner_utils.convert_to_metrics_dict(
                        results, self.oracle.objective, "Tuner.run_trial()"
                    ),
                )
            self.on_trial_end(trial)
        self.on_search_end()

    def run_trial(self, trial, *fit_args, **fit_kwargs):
        """Evaluates a set of hyperparameter values."""
        raise NotImplementedError

    def save_model(self, trial_id, model, step=0):
        """Saves a Model for a given trial.

        Args:
            trial_id: The ID of the `Trial` corresponding to this Model.
            model: The trained model.
            step: Integer, for models that report intermediate results to the
                `Oracle`, the step the saved file correspond to. For example, for
                Keras models this is the number of epochs trained.
        """
        raise NotImplementedError

    def load_model(self, trial):
        """Loads a Model from a given trial.

        For models that report intermediate results to the `Oracle`, generally
        `load_model` should load the best reported `step` by relying of
        `trial.best_step`.

        Args:
            trial: A `Trial` instance, the `Trial` corresponding to the model
                to load.
        """
        raise NotImplementedError

    def on_trial_begin(self, trial):
        """Called at the beginning of a trial.

        Args:
            trial: A `Trial` instance.
        """
        if self.logger:
            self.logger.register_trial(trial.trial_id, trial.get_state())
        self._display.on_trial_begin(self.oracle.get_trial(trial.trial_id))

    def on_trial_end(self, trial):
        """Called at the end of a trial.

        Args:
            trial: A `Trial` instance.
        """
        # Send status to Logger
        if self.logger:
            self.logger.report_trial_state(trial.trial_id, trial.get_state())

        self.oracle.end_trial(trial.trial_id, trial_module.TrialStatus.COMPLETED)
        self.oracle.update_space(trial.hyperparameters)
        # Display needs the updated trial scored by the Oracle.
        self._display.on_trial_end(self.oracle.get_trial(trial.trial_id))
        self.save()

    def on_search_begin(self):
        """Called at the beginning of the `search` method."""
        if self.logger:
            self.logger.register_tuner(self.get_state())

    def on_search_end(self):
        """Called at the end of the `search` method."""
        if self.logger:
            self.logger.exit()

    def get_best_models(self, num_models=1):
        """Returns the best model(s), as determined by the objective.

        This method is for querying the models trained during the search.
        For best performance, it is recommended to retrain your Model on the
        full dataset using the best hyperparameters found during `search`,
        which can be obtained using `tuner.get_best_hyperparameters()`.

        Args:
            num_models: Optional number of best models to return.
                Defaults to 1.

        Returns:
            List of trained models sorted from the best to the worst.
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

        Args:
            num_trials: Optional number of `HyperParameters` objects to return.

        Returns:
            List of `HyperParameter` objects sorted from the best to the worst.
        """
        return [t.hyperparameters for t in self.oracle.get_best_trials(num_trials)]

    def search_space_summary(self, extended=False):
        """Print search space summary.

        The methods prints a summary of the hyperparameters in the search
        space, which can be called before calling the `search` method.

        Args:
            extended: Optional boolean, whether to display an extended summary.
                Defaults to False.
        """
        print("Search space summary")
        hp = self.oracle.get_space()
        print("Default search space size: %d" % len(hp.space))
        for p in hp.space:
            config = p.get_config()
            name = config.pop("name")
            print("%s (%s)" % (name, p.__class__.__name__))
            print(config)

    def results_summary(self, num_trials=10):
        """Display tuning results summary.

        The method prints a summary of the search results including the
        hyperparameter values and evaluation results for each trial.

        Args:
            num_trials: Optional number of trials to display. Defaults to 10.
        """
        print("Results summary")
        print("Results in %s" % self.project_dir)
        print("Showing %d best trials" % num_trials)
        print("{}".format(self.oracle.objective))

        best_trials = self.oracle.get_best_trials(num_trials)
        for trial in best_trials:
            trial.summary()

    @property
    def remaining_trials(self):
        """Returns the number of trials remaining.

        Will return `None` if `max_trials` is not set. This is useful when
        resuming a previously stopped search.
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
        dirname = os.path.join(str(self.directory), self.project_name)
        utils.create_directory(dirname)
        return dirname

    def get_trial_dir(self, trial_id):
        dirname = os.path.join(str(self.project_dir), "trial_" + str(trial_id))
        utils.create_directory(dirname)
        return dirname

    def _get_tuner_fname(self):
        return os.path.join(str(self.project_dir), str(self.tuner_id) + ".json")
