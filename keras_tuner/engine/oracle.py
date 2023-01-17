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
"Oracle base class."


import collections
import hashlib
import json
import os
import random
import threading
import warnings

import numpy as np
import tensorflow as tf

from keras_tuner import utils
from keras_tuner.engine import hyperparameters as hp_module
from keras_tuner.engine import objective as obj_module
from keras_tuner.engine import stateful
from keras_tuner.engine import trial as trial_module

# For backward compatibility.
Objective = obj_module.Objective

# Map each `Oracle` instance to its `Lock`.
LOCKS = collections.defaultdict(lambda: threading.Lock())
# Map each `Oracle` instance to the thread name aquired the `Lock`.
THREADS = collections.defaultdict(lambda: None)


def synchronized(func, *args, **kwargs):
    """Decorator to synchronize the multi-threaded calls to a `Oracle` functions.

    In parallel tuning, there may be concurrent gRPC calls from multiple threads
    to the `Oracle` methods like `create_trial()`, `update_trial()`, and
    `end_trial()`. To avoid concurrent writing to the data, use `@synchronized`
    to ensure the calls are synchronized, which only allows one call to run at a
    time.

    Concurrent calls to different `Oracle` objects would not block one another.
    Concurrent calls to the same or different functions of the same `Oracle`
    object would block one another.

    You can decorate a subclass function, which overrides an already decorated
    function in the base class, without worrying about creating a deadlock.
    However, the decorator only support methods within classes, and cannot be
    applied to standalone functions.

    You do not need to decorate `Oracle.populate_space()`, which is only
    called by `Oracle.create_trial()`, which is decorated.

    Example:

    ```py
    class MyOracle(keras_tuner.Oracle):
        @keras_tuner.synchronized
        def create_trial(self, tuner_id):
            super().create_trial(tuner_id)
            ...

        @keras_tuner.synchronized
        def update_trial(self, trial_id, metrics, step=0):
            super().update_trial(trial_id, metrics, step)
            ...

        @keras_tuner.synchronized
        def end_trial(self, trial):
            super().end_trial(trial)
            ...
    ```
    """

    def backward_compatible_end_trial(self, trial_id, status):
        trial = trial_module.Trial(self.get_space(), trial_id, status)
        return [self, trial], {}

    def wrapped_func(*args, **kwargs):
        # For backward compatible with the old end_trial signature:
        # def end_trial(self, trial_id, status="COMPLETED"):
        if func.__name__ == "end_trial" and (
            "trial_id" in kwargs or "status" in kwargs or isinstance(args[1], str)
        ):
            args, kwargs = backward_compatible_end_trial(*args, **kwargs)

        oracle = args[0]
        thread_name = threading.currentThread().getName()
        need_acquire = THREADS[oracle] != thread_name

        if need_acquire:
            LOCKS[oracle].acquire()
            THREADS[oracle] = thread_name
        ret_val = func(*args, **kwargs)
        if need_acquire:
            THREADS[oracle] = None
            LOCKS[oracle].release()
        return ret_val

    return wrapped_func


class Oracle(stateful.Stateful):
    """Implements a hyperparameter optimization algorithm.

    In a parallel tuning setting, there is only one `Oracle` instance. The
    workers would communicate with the centralized `Oracle` instance with gPRC
    calls to the `Oracle` methods.

    `Trial` objects are often used as the communication packet through the gPRC
    calls to pass information between the worker `Tuner` instances and the
    `Oracle`. For example, `Oracle.create_trial()` returns a `Trial` object, and
    `Oracle.end_trial()` accepts a `Trial` in its arguments.

    New copies of the same `Trial` instance are reconstructed as it going
    through the gRPC calls. The changes to the `Trial` objects in the worker
    `Tuner`s are synced to the original copy in the `Oracle` when they are
    passed back to the `Oracle` by calling `Oracle.end_trial()`.

    Args:
        objective: A string, `keras_tuner.Objective` instance, or a list of
            `keras_tuner.Objective`s and strings. If a string, the direction of
            the optimization (min or max) will be inferred. If a list of
            `keras_tuner.Objective`, we will minimize the sum of all the
            objectives to minimize subtracting the sum of all the objectives to
            maximize. The `objective` argument is optional when
            `Tuner.run_trial()` or `HyperModel.fit()` returns a single float as
            the objective to minimize.
        max_trials: Integer, the total number of trials (model configurations)
            to test at most. Note that the oracle may interrupt the search
            before `max_trial` models have been tested if the search space has
            been exhausted.
        hyperparameters: Optional `HyperParameters` instance. Can be used to
            override (or register in advance) hyperparameters in the search
            space.
        tune_new_entries: Boolean, whether hyperparameter entries that are
            requested by the hypermodel but that were not specified in
            `hyperparameters` should be added to the search space, or not. If
            not, then the default value for these parameters will be used.
            Defaults to True.
        allow_new_entries: Boolean, whether the hypermodel is allowed to
            request hyperparameter entries not listed in `hyperparameters`.
            Defaults to True.
        seed: Int. Random seed.
        max_retries_per_trial: Integer. Defaults to 0. The maximum number of
            times to retry a `Trial` if the trial crashed or the results are invalid.
        max_consecutive_failed_trials: Integer. Defaults to 3. The maximum number of
            consecutive failed `Trial`s. When this number is reached, the search
            will be stopped. A `Trial` is marked as failed when none of the
            retries succeeded.
    """

    def __init__(
        self,
        objective=None,
        max_trials=None,
        hyperparameters=None,
        allow_new_entries=True,
        tune_new_entries=True,
        seed=None,
        max_retries_per_trial=0,
        max_consecutive_failed_trials=3,
    ):
        self.objective = obj_module.create_objective(objective)
        self.max_trials = max_trials
        if not hyperparameters:
            if not tune_new_entries:
                raise ValueError(
                    "If you set `tune_new_entries=False`, you must"
                    "specify the search space via the "
                    "`hyperparameters` argument."
                )
            if not allow_new_entries:
                raise ValueError(
                    "If you set `allow_new_entries=False`, you must"
                    "specify the search space via the "
                    "`hyperparameters` argument."
                )
            self.hyperparameters = hp_module.HyperParameters()
        else:
            self.hyperparameters = hyperparameters
        self.allow_new_entries = allow_new_entries
        self.tune_new_entries = tune_new_entries

        # trial_id -> Trial
        self.trials = {}
        # tuner_id -> Trial
        self.ongoing_trials = {}
        # List of trial_ids in the order of the trials start
        self.start_order = []
        # List of trial_ids in the order of the trials end
        self.end_order = []
        # Map trial_id to failed times
        self._run_times = collections.defaultdict(lambda: 0)
        # Used as a queue of trial_id to retry
        self._retry_queue = []

        self.seed = seed or random.randint(1, 10000)
        self._seed_state = self.seed
        # Hashes of values in the trials, which only hashes the active values.
        self._tried_so_far = set()
        # Dictionary mapping trial_id to the the hash of the values.
        self._id_to_hash = collections.defaultdict(lambda: None)
        # Maximum number of identical values that can be generated
        # before we consider the space to be exhausted.
        self._max_collisions = 20

        # Set in `BaseTuner` via `set_project_dir`.
        self.directory = None
        self.project_name = None

        # In multi-worker mode, only the chief of each cluster should report
        # results. These 2 attributes exist in `Oracle` just make it consistent
        # with `OracleClient`, in which the attributes are utilized.
        self.multi_worker = False
        self.should_report = True

        # Handling the retries and failed trials.
        self.max_retries_per_trial = max_retries_per_trial
        self.max_consecutive_failed_trials = max_consecutive_failed_trials

    def _populate_space(self, trial_id):
        warnings.warn(
            "The `_populate_space` method is deprecated, "
            "please use `populate_space`.",
            DeprecationWarning,
        )
        return self.populate_space(trial_id)

    def populate_space(self, trial_id):
        """Fill the hyperparameter space with values for a trial.

        This method should be overridden in subclasses and called in
        `create_trial` in order to populate the hyperparameter space with
        values.

        Args:
            trial_id: A string, the ID for this Trial.

        Returns:
            A dictionary with keys "values" and "status", where "values" is
            a mapping of parameter names to suggested values, and "status"
            should be one of "RUNNING" (the trial can start normally), "IDLE"
            (the oracle is waiting on something and cannot create a trial), or
            "STOPPED" (the oracle has finished searching and no new trial should
            be created).
        """
        raise NotImplementedError

    def _score_trial(self, trial):
        warnings.warn(
            "The `_score_trial` method is deprecated, please use `score_trial`.",
            DeprecationWarning,
        )
        self.score_trial(trial)

    def score_trial(self, trial):
        """Score a completed `Trial`.

        This method can be overridden in subclasses to provide a score for
        a set of hyperparameter values. This method is called from `end_trial`
        on completed `Trial`s.

        Args:
            trial: A completed `Trial` object.
        """
        trial.score = trial.metrics.get_best_value(self.objective.name)
        trial.best_step = trial.metrics.get_best_step(self.objective.name)

    @synchronized
    def create_trial(self, tuner_id):
        """Create a new `Trial` to be run by the `Tuner`.

        A `Trial` corresponds to a unique set of hyperparameters to be run
        by `Tuner.run_trial`.

        Args:
            tuner_id: A string, the ID that identifies the `Tuner` requesting a
                `Trial`. `Tuners` that should run the same trial (for instance,
                when running a multi-worker model) should have the same ID.

        Returns:
            A `Trial` object containing a set of hyperparameter values to run
            in a `Tuner`.
        """
        # Allow for multi-worker DistributionStrategy within a Trial.
        if tuner_id in self.ongoing_trials:
            return self.ongoing_trials[tuner_id]

        # Pick the Trials waiting for retry first.
        if len(self._retry_queue) > 0:
            trial = self.trials[self._retry_queue.pop()]
            trial.status = trial_module.TrialStatus.RUNNING
            self.ongoing_trials[tuner_id] = trial
            self.save()
            return trial

        # Make the trial_id the current number of trial, pre-padded with 0s
        trial_id = f"{{:0{len(str(self.max_trials))}d}}"
        trial_id = trial_id.format(len(self.trials))

        if self.max_trials and len(self.trials) >= self.max_trials:
            status = trial_module.TrialStatus.STOPPED
            values = None
        else:
            response = self.populate_space(trial_id)
            status = response["status"]
            values = response["values"] if "values" in response else None

        hyperparameters = self.hyperparameters.copy()
        hyperparameters.values = values or {}

        trial = trial_module.Trial(
            hyperparameters=hyperparameters, trial_id=trial_id, status=status
        )

        if status == trial_module.TrialStatus.RUNNING:
            # Record the populated values (active only). Only record when the
            # status is RUNNING. If other status, the trial will not run, the
            # values are discarded and should not be recorded, in which case,
            # the trial_id may appear again in the future.
            self._record_values(trial)

            self.ongoing_trials[tuner_id] = trial
            self.trials[trial_id] = trial
            self.start_order.append(trial_id)
            self._save_trial(trial)
            self.save()

        return trial

    @synchronized
    def update_trial(self, trial_id, metrics, step=0):
        """Used by a worker to report the status of a trial.

        Args:
            trial_id: A string, a previously seen trial id.
            metrics: Dict. The keys are metric names, and the values are this
                trial's metric values.
            step: Optional float, reporting intermediate results. The current
                value in a timeseries representing the state of the trial. This
                is the value that `metrics` will be associated with.

        Returns:
            Trial object.
        """
        trial = self.trials[trial_id]
        self._check_objective_found(metrics)
        for metric_name, metric_value in metrics.items():
            if not trial.metrics.exists(metric_name):
                direction = _maybe_infer_direction_from_objective(
                    self.objective, metric_name
                )
                trial.metrics.register(metric_name, direction=direction)
            trial.metrics.update(metric_name, metric_value, step=step)
        self._save_trial(trial)
        # TODO: To signal early stopping, set Trial.status to "STOPPED".
        return trial.status

    def _check_consecutive_failures(self):
        # For thread safety, check all trials for consecutive failures.
        consecutive_failures = 0
        for trial_id in self.end_order:
            trial = self.trials[trial_id]
            if trial.status == trial_module.TrialStatus.FAILED:
                consecutive_failures += 1
            else:
                consecutive_failures = 0
            if consecutive_failures == self.max_consecutive_failed_trials:
                raise RuntimeError(
                    "Number of consecutive failures excceeded the limit "
                    f"of {self.max_consecutive_failed_trials}.\n" + trial.message
                )

    @synchronized
    def end_trial(self, trial):
        """Logistics when a `Trial` finished running.

        Record the `Trial` information and end the trial or send it for retry.

        Args:
            trial: The Trial to be ended. `trial.status` should be one of
                `"COMPLETED"` (the trial finished normally), `"INVALID"` (the
                trial has crashed or been deemed infeasible, but subject to
                retries), or `"FAILED"` (The Trial is failed. No more retries
                needed.). `trial.message` is an optional string, which is the
                error message if the trial status is `"INVALID"` or `"FAILED"`.
        """
        for tuner_id, ongoing_trial in self.ongoing_trials.items():
            if ongoing_trial.trial_id == trial.trial_id:
                self.ongoing_trials.pop(tuner_id)
                break

        # To support parallel tuning, the information in the `trial` argument is
        # synced back to the `Oracle`. Update the self.trials with the given
        # trial.
        old_trial = self.trials[trial.trial_id]
        old_trial.hyperparameters = trial.hyperparameters
        old_trial.status = trial.status
        old_trial.message = trial.message
        trial = old_trial

        self.update_space(trial.hyperparameters)
        if trial.status == trial_module.TrialStatus.COMPLETED:
            self.score_trial(trial)
            if np.isnan(trial.score):
                trial.status = trial_module.TrialStatus.INVALID

        # Record the values again in case of new hps appeared.
        self._record_values(trial)

        self._run_times[trial.trial_id] += 1

        # Check if need to retry the trial.
        if not self._retry(trial):
            self.end_order.append(trial.trial_id)
            self._check_consecutive_failures()

        self._save_trial(trial)
        self.save()

    def _retry(self, trial):
        """Send the trial for retry if needed.

        Args:
            trial: Trial. The trial to check.

        Returns:
            Boolean. Whether the trial should be retried.
        """
        if trial.status != trial_module.TrialStatus.INVALID:
            return False

        trial_id = trial.trial_id
        max_run_times = self.max_retries_per_trial + 1

        if self._run_times[trial_id] >= max_run_times:
            trial.status = trial_module.TrialStatus.FAILED
            return False

        print(
            f"Trial {trial_id} failed {self._run_times[trial_id]} "
            "times. "
            f"{max_run_times - self._run_times[trial_id]} "
            "retries left."
        )
        self._retry_queue.append(trial_id)
        return True

    def get_space(self):
        """Returns the `HyperParameters` search space."""
        return self.hyperparameters.copy()

    def update_space(self, hyperparameters):
        """Add new hyperparameters to the tracking space.

        Already recorded parameters get ignored.

        Args:
            hyperparameters: An updated `HyperParameters` object.
        """
        hps = hyperparameters.space
        new_hps = [
            hp
            for hp in hps
            if not self.hyperparameters._exists(hp.name, hp.conditions)
        ]

        if new_hps and not self.allow_new_entries:
            raise RuntimeError(
                f"`allow_new_entries` is `False`, but found new entries {new_hps}"
            )
        if not self.tune_new_entries:
            # New entries should always use the default value.
            return
        self.hyperparameters.merge(new_hps)

    def get_trial(self, trial_id):
        """Returns the `Trial` specified by `trial_id`."""
        return self.trials[trial_id]

    def get_best_trials(self, num_trials=1):
        """Returns the best `Trial`s."""
        trials = [
            t
            for t in self.trials.values()
            if t.status == trial_module.TrialStatus.COMPLETED
        ]

        sorted_trials = sorted(
            trials,
            key=lambda trial: trial.score,
            reverse=self.objective.direction == "max",
        )
        return sorted_trials[:num_trials]

    def remaining_trials(self):
        return (
            self.max_trials - len(self.trials.items()) if self.max_trials else None
        )

    def get_state(self):
        # `self.trials` are saved in their own, Oracle-agnostic files.
        # Just save the IDs for ongoing trials, since these are in `trials`.
        return {
            "ongoing_trials": {
                tuner_id: trial.trial_id
                for tuner_id, trial in self.ongoing_trials.items()
            },
            # Hyperparameters are part of the state because they can be added to
            # during the course of the search.
            "hyperparameters": self.hyperparameters.get_config(),
            "start_order": self.start_order,
            "end_order": self.end_order,
            "run_times": self._run_times,
            "retry_queue": self._retry_queue,
            "seed": self.seed,
            "seed_state": self._seed_state,
            "tried_so_far": list(self._tried_so_far),
            "id_to_hash": self._id_to_hash,
        }

    def set_state(self, state):
        # `self.trials` are saved in their own, Oracle-agnostic files.
        self.ongoing_trials = {
            tuner_id: self.trials[trial_id]
            for tuner_id, trial_id in state["ongoing_trials"].items()
        }
        self.hyperparameters = hp_module.HyperParameters.from_config(
            state["hyperparameters"]
        )
        self.start_order = state["start_order"]
        self.end_order = state["end_order"]
        self._run_times = collections.defaultdict(lambda: 0)
        self._run_times.update(state["run_times"])
        self._retry_queue = state["retry_queue"]
        self.seed = state["seed"]
        self._seed_state = state["seed_state"]
        self._tried_so_far = set(state["tried_so_far"])
        self._id_to_hash = collections.defaultdict(lambda: None)
        self._id_to_hash.update(state["id_to_hash"])

    def _set_project_dir(self, directory, project_name):
        """Sets the project directory and reloads the Oracle."""
        self._directory = directory
        self._project_name = project_name

    @property
    def _project_dir(self):
        dirname = os.path.join(str(self._directory), self._project_name)
        utils.create_directory(dirname)
        return dirname

    def save(self):
        # `self.trials` are saved in their own, Oracle-agnostic files.
        super().save(self._get_oracle_fname())

    def reload(self):
        # Reload trials from their own files.
        trial_fnames = tf.io.gfile.glob(
            os.path.join(self._project_dir, "trial_*", "trial.json")
        )
        for fname in trial_fnames:
            with tf.io.gfile.GFile(fname, "r") as f:
                trial_data = f.read()
            trial_state = json.loads(trial_data)
            trial = trial_module.Trial.from_state(trial_state)
            self.trials[trial.trial_id] = trial
        try:
            super().reload(self._get_oracle_fname())
        except KeyError as e:
            raise RuntimeError(
                "Error reloading `Oracle` from existing project. "
                "If you did not mean to reload from an existing project, "
                f"change the `project_name` or pass `overwrite=True` "
                "when creating the `Tuner`. Found existing "
                f"project at: {self._project_dir}"
            ) from e

        # Empty the ongoing_trials and send them for retry.
        for _, trial_id in self.ongoing_trials.items():
            self._retry_queue.append(trial_id)
        self.ongoing_trials = {}

    def _get_oracle_fname(self):
        return os.path.join(self._project_dir, "oracle.json")

    def _compute_values_hash(self, values):
        keys = sorted(values.keys())
        s = "".join(f"{str(k)}={str(values[k])}" for k in keys)
        return hashlib.sha256(s.encode("utf-8")).hexdigest()[:32]

    def _check_objective_found(self, metrics):
        if isinstance(self.objective, obj_module.Objective):
            objective_names = [self.objective.name]
        else:
            objective_names = [obj.name for obj in self.objective]
        for metric_name in metrics.keys():
            if metric_name in objective_names:
                objective_names.remove(metric_name)
        if objective_names:
            raise ValueError(
                "Objective value missing in metrics reported to "
                f"the Oracle, expected: {objective_names}, "
                f"found: {metrics.keys()}"
            )

    def _get_trial_dir(self, trial_id):
        dirname = os.path.join(self._project_dir, f"trial_{str(trial_id)}")
        utils.create_directory(dirname)
        return dirname

    def _save_trial(self, trial):
        # Write trial status to trial directory
        trial_id = trial.trial_id
        trial.save(os.path.join(self._get_trial_dir(trial_id), "trial.json"))

    def _random_values(self):
        """Fills the hyperparameter space with random values.

        Returns:
            A dictionary mapping hyperparameter names to suggested values.
        """
        collisions = 0
        while 1:
            hps = hp_module.HyperParameters()
            # Generate a set of random values.
            for hp in self.hyperparameters.space:
                hps.merge([hp])
                if hps.is_active(hp):  # Only active params in `values`.
                    hps.values[hp.name] = hp.random_sample(self._seed_state)
                    self._seed_state += 1
            # Keep trying until the set of values is unique,
            # or until we exit due to too many collisions.
            if self._duplicate(hps.values):
                collisions += 1
                if collisions > self._max_collisions:
                    return None
                continue
            break
        return hps.values

    def _duplicate(self, values):
        """Check if the values has been tried in previous trials.

        Args:
            A dictionary mapping hyperparameter names to suggested values.

        Returns:
            Boolean. Whether the values has been tried in previous trials.
        """
        return self._compute_values_hash(values) in self._tried_so_far

    def _record_values(self, trial):
        hyperparameters = trial.hyperparameters
        hyperparameters.ensure_active_values()
        new_hash_value = self._compute_values_hash(hyperparameters.values)
        self._tried_so_far.add(new_hash_value)

        # In case of new hp appeared, remove the old hash value.
        old_hash_value = self._id_to_hash[trial.trial_id]
        if old_hash_value is None:
            self._id_to_hash[trial.trial_id] = new_hash_value
        elif old_hash_value != new_hash_value:
            self._tried_so_far.remove(old_hash_value)


def _maybe_infer_direction_from_objective(objective, metric_name):
    if isinstance(objective, obj_module.Objective):
        objective = [objective]
    return next(
        (obj.direction for obj in objective if obj.name == metric_name), None
    )
