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
        if len(self.start_order) > 0:
            last_trial = self.trials[self.start_order[-1]]
            last_values = last_trial.hyperparameters.values
            values = self._get_next_combination(last_values)
        else:
            # Use all default values for the first trial.
            values = {hp.name: hp.default for hp in self.get_space().space}
        if values is None:
            return {"status": trial_module.TrialStatus.STOPPED, "values": None}
        return {"status": trial_module.TrialStatus.RUNNING, "values": values}

    def _get_next_combination(self, values):
        """Get the next value combination.

        This function is called on the second and later trials, but not on the
        first one. Even new hps appear during search, the all hps in
        self.get_space() should have corresponding values in the `values` arg.
        This is because `values` is from the last trial, whose values has been
        updated during that trial, which contains all seen hps.

        We treat the default value of a hp as the first value, and the rest are
        in normal order. We will enumerate the values according to this order.
        Get all possible hyperparameter values
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
        names = [hp.name for hp in hps.space]  # Ordered
        new_values = copy.deepcopy(values)

        bumped_value = False

        # Iterate in reverse order so that we can change the value under
        # conditional scope first instead of change the condition value first.
        for name in reversed(names):
            # Bump up the hp value if possible.
            if new_values[name] != all_values[name][-1]:
                index = all_values[name].index(new_values[name]) + 1
                new_values[name] = all_values[name][index]
                bumped_value = True
                break
            # Otherwise, reset to its first value.
            new_values[name] = default_values[name]

        if bumped_value:
            return new_values

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

    For the following hyperparameter types, GridSearch will not exhaust all
    possible values, but pick 10 samples evenly.
    * `hp.Float()`.
    * `hp.Int()` with `sampling` set to `"log"` or `"reverse_log"`.

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
