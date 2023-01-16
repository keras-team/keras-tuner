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
import traceback
import warnings

import tensorflow as tf

from keras_tuner import config as config_module
from keras_tuner import errors
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

    `BaseTuner` is the super class of all `Tuner` classes. It defines the APIs
    for the `Tuner` classes and serves as a wrapper class for the internal
    logics.

    `BaseTuner` supports parallel tuning. In parallel tuning, the communication
    between `BaseTuner` and `Oracle` are all going through gRPC. There are
    multiple running instances of `BaseTuner` but only one `Oracle`. This design
    allows the user to run the same script on multiple machines to launch the
    parallel tuning.

    The `Oracle` instance should manage the life cycles of all the `Trial`s,
    while a `BaseTuner` is a worker for running the `Trial`s. `BaseTuner`s
    requests `Trial`s from the `Oracle`, run them, and report the results back
    to the `Oracle`. A `BaseTuner` also handles events happening during running
    the `Trial`, like saving the model, logging, error handling. Other than
    these responsibilities, a `BaseTuner` should avoid managing a `Trial` since
    the relevant contexts for a `Trial` are in the `Oracle`, which only
    accessible from gRPC.

    The `BaseTuner` should be a general tuner for all types of models and avoid
    any logic directly related to Keras. The Keras related logics should be
    handled by the `Tuner` class, which is a subclass of `BaseTuner`.

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
        if not isinstance(oracle, oracle_module.Oracle):
            raise ValueError(
                "Expected `oracle` argument to be an instance of `Oracle`. "
                f"Received: oracle={oracle} (of type ({type(oracle)})."
            )

        self.oracle = oracle
        self.hypermodel = hm_module.get_hypermodel(hypermodel)

        # Ops and metadata
        self.directory = directory or "."
        self.project_name = project_name or "untitled_project"
        self.oracle._set_project_dir(self.directory, self.project_name)

        self.logger = logger

        if overwrite and tf.io.gfile.exists(self.project_dir):
            tf.io.gfile.rmtree(self.project_dir)

        # To support tuning distribution.
        self.tuner_id = os.environ.get("KERASTUNER_TUNER_ID", "tuner0")

        # Reloading state.
        if not overwrite and tf.io.gfile.exists(self._get_tuner_fname()):
            tf.get_logger().info(f"Reloading Tuner from {self._get_tuner_fname()}")
            self.reload()
        else:
            # Only populate initial space if not reloading.
            self._populate_initial_space()

        # Run in distributed mode.
        if dist_utils.is_chief_oracle():
            # Blocks forever.
            oracle_chief.start_server(self.oracle)
        elif dist_utils.has_chief_oracle():
            # Proxies requests to the chief oracle.
            self.oracle = oracle_client.OracleClient(self.oracle)

        # In parallel tuning, everything below in __init__() is for workers only.
        # Logs etc
        self._display = tuner_utils.Display(oracle=self.oracle)

    def _activate_all_conditions(self):
        # Lists of stacks of conditions used during `explore_space()`.
        scopes_never_active = []
        scopes_once_active = []

        hp = self.oracle.get_space()
        while True:
            self.hypermodel.build(hp)
            self.oracle.update_space(hp)

            # Update the recorded scopes.
            for conditions in hp.active_scopes:
                if conditions not in scopes_once_active:
                    scopes_once_active.append(copy.deepcopy(conditions))
                if conditions in scopes_never_active:
                    scopes_never_active.remove(conditions)
            for conditions in hp.inactive_scopes:
                if conditions not in scopes_once_active:
                    scopes_never_active.append(copy.deepcopy(conditions))

            # All conditional scopes are activated.
            if not scopes_never_active:
                break

            # Generate new values to activate new conditions.
            hp = self.oracle.get_space()
            conditions = scopes_never_active[0]
            for condition in conditions:
                hp.values[condition.name] = condition.values[0]

            hp.ensure_active_values()

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

        # declare_hyperparameters is not overriden.
        hp = self.oracle.get_space()
        self.hypermodel.declare_hyperparameters(hp)
        self.oracle.update_space(hp)
        self._activate_all_conditions()

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
            self.pre_create_trial()
            trial = self.oracle.create_trial(self.tuner_id)
            if trial.status == trial_module.TrialStatus.STOPPED:
                # Oracle triggered exit.
                tf.get_logger().info("Oracle triggered exit")
                break
            if trial.status == trial_module.TrialStatus.IDLE:
                # Oracle is calculating, resend request.
                continue

            self.on_trial_begin(trial)
            self._try_run_and_update_trial(trial, *fit_args, **fit_kwargs)
            self.on_trial_end(trial)
        self.on_search_end()

    def _run_and_update_trial(self, trial, *fit_args, **fit_kwargs):
        results = self.run_trial(trial, *fit_args, **fit_kwargs)
        if self.oracle.get_trial(trial.trial_id).metrics.exists(
            self.oracle.objective.name
        ):
            # The oracle is updated by calling `self.oracle.update_trial()` in
            # `Tuner.run_trial()`. For backward compatibility, we support this
            # use case. No further action needed in this case.
            warnings.warn(
                "The use case of calling "
                "`self.oracle.update_trial(trial_id, metrics)` "
                "in `Tuner.run_trial()` to report the metrics is deprecated, "
                "and will be removed in the future."
                "Please remove the call and do 'return metrics' "
                "in `Tuner.run_trial()` instead. ",
                DeprecationWarning,
                stacklevel=2,
            )
            return

        tuner_utils.validate_trial_results(
            results, self.oracle.objective, "Tuner.run_trial()"
        ),
        self.oracle.update_trial(
            trial.trial_id,
            # Convert to dictionary before calling `update_trial()`
            # to pass it from gRPC.
            tuner_utils.convert_to_metrics_dict(
                results,
                self.oracle.objective,
            ),
            step=tuner_utils.get_best_step(results, self.oracle.objective),
        )

    def _try_run_and_update_trial(self, trial, *fit_args, **fit_kwargs):
        try:
            self._run_and_update_trial(trial, *fit_args, **fit_kwargs)
            trial.status = trial_module.TrialStatus.COMPLETED
            return
        except Exception as e:
            if isinstance(e, errors.FatalError):
                raise e
            if config_module.DEBUG:
                # Printing the stacktrace and the error.
                traceback.print_exc()

            if isinstance(e, errors.FailedTrialError):
                trial.status = trial_module.TrialStatus.FAILED
            else:
                trial.status = trial_module.TrialStatus.INVALID

            # Include the stack traces in the message.
            message = traceback.format_exc()
            trial.message = message

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

    def pre_create_trial(self):
        """Called before self.oracle.create_trial and before on_trial_begin."""

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

        self.oracle.end_trial(trial)
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
        print(f"Default search space size: {len(hp.space)}")
        for p in hp.space:
            config = p.get_config()
            name = config.pop("name")
            print(f"{name} ({p.__class__.__name__})")
            print(config)

    def results_summary(self, num_trials=10):
        """Display tuning results summary.

        The method prints a summary of the search results including the
        hyperparameter values and evaluation results for each trial.

        Args:
            num_trials: Optional number of trials to display. Defaults to 10.
        """
        print("Results summary")
        print(f"Results in {self.project_dir}")
        print("Showing %d best trials" % num_trials)
        print(f"{self.oracle.objective}")

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

    def _is_worker(self):
        """Return true only if in parallel tuning and is a worker tuner."""
        return dist_utils.has_chief_oracle() and not dist_utils.is_chief_oracle()

    def save(self):
        """Saves this object to its project directory."""
        if not self._is_worker():
            self.oracle.save()
        super().save(self._get_tuner_fname())

    def reload(self):
        """Reloads this object from its project directory."""
        if not self._is_worker():
            self.oracle.reload()
        super().reload(self._get_tuner_fname())

    @property
    def project_dir(self):
        dirname = os.path.join(str(self.directory), self.project_name)
        utils.create_directory(dirname)
        return dirname

    def get_trial_dir(self, trial_id):
        dirname = os.path.join(str(self.project_dir), f"trial_{str(trial_id)}")
        utils.create_directory(dirname)
        return dirname

    def _get_tuner_fname(self):
        return os.path.join(str(self.project_dir), f"{str(self.tuner_id)}.json")
