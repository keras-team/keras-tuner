# Copyright 2022 The KerasTuner Authors
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

"Basic exhaustive search tuner."


from keras_tuner.engine import hyperparameters as hp_module
from keras_tuner.engine import oracle as oracle_module
from keras_tuner.engine import trial as trial_lib
from keras_tuner.engine import trial as trial_module
from keras_tuner.engine import tuner as tuner_module


class GridSearchOracle(oracle_module.Oracle):
    """Grid search oracle.

    Args:
        objective: A string, `keras_tuner.Objective` instance, or a list of
            `keras_tuner.Objective`s and strings. If a string, the direction of
            the optimization (min or max) will be inferred. If a list of
            `keras_tuner.Objective`, we will minimize the sum of all the
            objectives to minimize subtracting the sum of all the objectives to
            maximize. The `objective` argument is optional when
            `Tuner.run_trial()` or `HyperModel.fit()` returns a single float as
            the objective to minimize.
        max_trials: Optional integer, the total number of trials (model
            configurations) to test at most. Note that the oracle may interrupt
            the search before `max_trial` models have been tested if the search
            space has been exhausted. If left unspecified, it runs till the
            search space is exhausted.
        seed: Optional integer, the random seed.
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
    """

    def __init__(
        self,
        objective=None,
        max_trials=None,
        seed=None,
        hyperparameters=None,
        allow_new_entries=True,
        tune_new_entries=True,
    ):
        super(GridSearchOracle, self).__init__(
            objective=objective,
            max_trials=max_trials,
            hyperparameters=hyperparameters,
            tune_new_entries=tune_new_entries,
            allow_new_entries=allow_new_entries,
            seed=seed,
        )

    def populate_space(self, trial_id):
        """Fill the hyperparameter space with values.

        Args:
            trial_id: A string, the ID for this Trial.

        Returns:
            A dictionary with keys "values" and "status", where "values" is
            a mapping of parameter names to suggested values, and "status"
            is the TrialStatus that should be returned for this trial (one
            of "RUNNING", "IDLE", or "STOPPED").
        """
        values = self._exhaustive_values()
        if values is None:
            return {"status": trial_module.TrialStatus.STOPPED, "values": None}
        return {"status": trial_module.TrialStatus.RUNNING, "values": values}

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

        # The hyperparameter types that allow an exhaustive search due to their
        # finite sets.
        allowed_types = {hp_module.Choice, hp_module.Boolean, hp_module.Fixed}
        # Calculates the size of the set of all possible choices.
        number_of_hp_choices = 1
        for hp in self.hyperparameters.space:
            if type(hp) not in allowed_types:
                raise ValueError(
                    "ExhaustiveSearch tuner accepts hyperparameters of type "
                    "Choice, Boolean and Fixed as it needs to have a finite "
                    f"set of choices to search, found: {hp}"
                )
            # Fixed has no "values" (it is actually 1) attribute as they do
            # not dispose of possible choices, so we skip counting it.
            # As its count is 1 it won't change the final product.
            # Boolean has True and False, count as 2.
            # Choice has the values given by its cardinality.
            if type(hp) == hp_module.Fixed:
                continue
            elif type(hp) == hp_module.Boolean:
                number_of_hp_choices = number_of_hp_choices * 2
            else:
                number_of_hp_choices = number_of_hp_choices * len(hp.values)

        # If max_trials was not given, take the calculated number of hp choices to exhaustion.
        # If max_trials was given, take the minimum value between it and the calculated number
        # of hp choices to exhaustion. It implies that grid search will run trials up to the maximum
        # amount of choices available in the grid, even if the given max_trials is a bigger value.
        self.max_trials = min(
            trial_count
            for trial_count in [number_of_hp_choices, self.max_trials]
            if trial_count is not None
        )

        # Make the trial_id the current number of trial, pre-padded with 0s
        trial_id = "{{:0{}d}}".format(len(str(self.max_trials)))
        trial_id = trial_id.format(len(self.trials))

        if self.max_trials and len(self.trials) >= self.max_trials:
            status = trial_lib.TrialStatus.STOPPED
            values = None
        else:
            response = self.populate_space(trial_id)
            status = response["status"]
            values = response["values"] if "values" in response else None

        hyperparameters = self.hyperparameters.copy()
        hyperparameters.values = values or {}
        trial = trial_lib.Trial(
            hyperparameters=hyperparameters, trial_id=trial_id, status=status
        )

        if status == trial_lib.TrialStatus.RUNNING:
            self.ongoing_trials[tuner_id] = trial
            self.trials[trial_id] = trial
            self.start_order.append(trial_id)
            self._save_trial(trial)
            self.save()

        return trial

    def _exhaustive_values(self):
        """Fills the hyperparameter space until exhaustion of values.

        Returns:
            A dictionary mapping parameter names to suggested values.
        """
        # Keep trying until have tried all the possible choice
        # combinations.
        while len(self._tried_so_far) < self.max_trials:
            hps = hp_module.HyperParameters()
            # Generate a set of random values.
            for hp in self.hyperparameters.space:
                hps.merge([hp])
                if hps.is_active(hp):  # Only active params in `values`.
                    hps.values[hp.name] = hp.random_sample(self._seed_state)
                    self._seed_state += 1
            values = hps.values
            values_hash = self._compute_values_hash(values)
            if values_hash in self._tried_so_far:
                continue
            self._tried_so_far.add(values_hash)
            return values
        return None


class GridSearch(tuner_module.Tuner):
    """The Grid search tuner.

    It will try all the possible hyperparameter
    combinations up to exhaustion.

    For example:

    ```py
    optimizer = hp.Choice("model_name", values=["sgd", "adam"])
    lr = hp.Choice("lr", values=[0.01, 0.1])
    ```

    This tuner will fit models for: ["sgd", 0.01], ["sgd", 0.1], ["adam", 0.01]
    ["adam", 0.1].

    Args:
        hypermodel: Instance of `HyperModel` class (or callable that takes
            hyperparameters and returns a Model instance). It is optional when
            `Tuner.run_trial()` is overridden and does not use
            `self.hypermodel`.
        objective: A string, `keras_tuner.Objective` instance, or a list of
            `keras_tuner.Objective`s and strings. If a string, the direction of
            the optimization (min or max) will be inferred. If a list of
            `keras_tuner.Objective`, we will minimize the sum of all the
            objectives to minimize subtracting the sum of all the objectives to
            maximize. The `objective` argument is optional when
            `Tuner.run_trial()` or `HyperModel.fit()` returns a single float as
            the objective to minimize.
        max_trials: Optional integer, the total number of trials (model
            configurations) to test at most. Note that the oracle may interrupt
            the search before `max_trial` models have been tested if the search
            space has been exhausted. If left unspecified, it runs till the
            search space is exhausted.
        seed: Optional integer, the random seed.
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
        **kwargs: Keyword arguments relevant to all `Tuner` subclasses.
            Please see the docstring for `Tuner`.
    """

    def __init__(
        self,
        hypermodel=None,
        objective=None,
        seed=None,
        hyperparameters=None,
        tune_new_entries=True,
        allow_new_entries=True,
        **kwargs,
    ):
        self.seed = seed
        oracle = GridSearchOracle(
            objective=objective,
            seed=seed,
            hyperparameters=hyperparameters,
            tune_new_entries=tune_new_entries,
            allow_new_entries=allow_new_entries,
        )
        super(GridSearch, self).__init__(oracle, hypermodel, **kwargs)
