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

"Basic exhaustive search tuner."


import copy

from keras_tuner.engine import oracle as oracle_module
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
        seed=None,
        hyperparameters=None,
        allow_new_entries=True,
        tune_new_entries=True,
        max_retries_per_trial=0,
        max_consecutive_failed_trials=3,
    ):
        super(GridSearchOracle, self).__init__(
            objective=objective,
            max_trials=max_trials,
            hyperparameters=hyperparameters,
            tune_new_entries=tune_new_entries,
            allow_new_entries=allow_new_entries,
            seed=seed,
            max_retries_per_trial=max_retries_per_trial,
            max_consecutive_failed_trials=max_consecutive_failed_trials,
        )

    def populate_space(self, trial_id):
        """Fill the hyperparameter space with values.

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
        if len(self.start_order) > 0:
            last_trial = self.trials[self.start_order[-1]]
            last_values = last_trial.hyperparameters.values
            # The keys (hp names) in the `last_values` are always consistent with
            # the hps in `self.get_space().space`, even for newly appeared hps.
            # For example, during last trial's `_populate_space()`, a new hp
            # has not appeared. During last trial's `HyperModel.build()`, the
            # new hp appeared. The `hyperparameters.values` is then updated
            # immediately.
            values = self._get_next_combination(last_values)
        else:
            # Use all default values for the first trial.
            values = {hp.name: hp.default for hp in self.get_space().space}
        if values is None:
            return {"status": trial_module.TrialStatus.STOPPED, "values": None}
        return {"status": trial_module.TrialStatus.RUNNING, "values": values}

    def _get_next_combination(self, values):
        """Get the next value combination to try.

        Given the last trial's values dictionary, this method retrieves the next
        hyperparameter values to try. As it requires the last trial's
        values as input, it should not be called on the first trial. The first
        trial will always use default hp values.

        This oracle iterates over the search space entirely deterministically.

        When a new hp appears in a trial, the first value tried for that hp
        will be its default value.

        Args:
            values: Dict. The keys are hp names. The values are the hp values
                from the last trial.

        Returns:
            Dict. The next possible value combination for the hyperparameters.
        """

        hps = self.get_space()
        all_values = {}
        for hp in hps.space:
            value_list = list(hp.values)
            if hp.default in value_list:
                value_list.remove(hp.default)
            # Put the default value first.
            all_values[hp.name] = [hp.default] + value_list
        default_values = {hp.name: hp.default for hp in hps.space}
        new_values = copy.deepcopy(values)
        hps.values = new_values

        bumped_value = False

        # Iterate in reverse order so that we can change the value under
        # conditional scope first instead of change the condition value first.
        for hp in reversed(hps.space):
            name = hp.name
            # Bump up the hp value if possible and active.
            if hps.is_active(hp):
                value = new_values[name]
                if value != all_values[name][-1]:
                    index = all_values[name].index(value) + 1
                    new_values[name] = all_values[name][index]
                    bumped_value = True
                    break
            # Otherwise, reset to its first value.
            new_values[name] = default_values[name]

        return new_values if bumped_value else None


class GridSearch(tuner_module.Tuner):
    """The grid search tuner.

    This tuner iterates over all possible
    hyperparameter combinations.

    For example, with:

    ```py
    optimizer = hp.Choice("model_name", values=["sgd", "adam"])
    learning_rate = hp.Choice("learning_rate", values=[0.01, 0.1])
    ```

    This tuner will cover the following combinations:
    `["sgd", 0.01], ["sgd", 0.1], ["adam", 0.01] ["adam", 0.1]`.

    For the following hyperparameter types, GridSearch will not exhaust all
    possible values:

    * `hp.Float()` when `step` is left unspecified.
    * `hp.Int()` with `sampling` set to `"log"` or `"reverse_log"`, and `step`
        is left unspecified.

    For these cases, KerasTuner will pick 10 samples in the range evenly by default.
    To configure the granularity of sampling for `hp.Float()` and `hp.Int()`,
    please use the `step` argument in their initializers.

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
        max_retries_per_trial: Integer. Defaults to 0. The maximum number of
            times to retry a `Trial` if the trial crashed or the results are invalid.
        max_consecutive_failed_trials: Integer. Defaults to 3. The maximum number of
            consecutive failed `Trial`s. When this number is reached, the search
            will be stopped. A `Trial` is marked as failed when none of the
            retries succeeded.
        **kwargs: Keyword arguments relevant to all `Tuner` subclasses.
            Please see the docstring for `Tuner`.
    """

    def __init__(
        self,
        hypermodel=None,
        objective=None,
        max_trials=None,
        seed=None,
        hyperparameters=None,
        tune_new_entries=True,
        allow_new_entries=True,
        max_retries_per_trial=0,
        max_consecutive_failed_trials=3,
        **kwargs,
    ):
        self.seed = seed
        oracle = GridSearchOracle(
            objective=objective,
            max_trials=max_trials,
            seed=seed,
            hyperparameters=hyperparameters,
            tune_new_entries=tune_new_entries,
            allow_new_entries=allow_new_entries,
            max_retries_per_trial=max_retries_per_trial,
            max_consecutive_failed_trials=max_consecutive_failed_trials,
        )
        super(GridSearch, self).__init__(oracle, hypermodel, **kwargs)
