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


import collections
import copy

from keras_tuner.engine import oracle as oracle_module
from keras_tuner.engine import trial as trial_module
from keras_tuner.engine import tuner as tuner_module


class LinkedList:
    """A simplified linked list with limited supported operations.

    It doesn't copy any data pass to it but directly refer to it.
    """

    def __init__(self):
        # _memory is a list to store data.
        # Its index is the address for the linked list.
        # index to data
        self._memory = []
        self._data_to_index = {}
        # index to index
        self._next_index = collections.defaultdict(lambda: None)
        self._last_index = None

    def insert(self, data, data_pre=None):
        """Insert data after another data.

        `data` is inserted after `data_pre` in the linked list.

        Args:
            data: The data to insert.
            data_pre: Optional. The data marking the insertion location. If left
                unspecified, the data will be appended to the rear of the linked
                list.
        """
        self._memory.append(data)
        new_index = len(self._memory) - 1
        self._data_to_index[data] = new_index

        index = (
            self._last_index if data_pre is None else self._data_to_index[data_pre]
        )

        self._next_index[new_index] = self._next_index[index]
        self._next_index[index] = new_index

        # Update self._last_index.
        while self._next_index[self._last_index] is not None:
            self._last_index = self._next_index[self._last_index]

    def next(self, data):
        """Get the next data for a given data.

        Args:
            data: The data used to get its next data in the linked list.

        Returns:
            The next data if exists. Otherwise, return None.
        """
        index = self._data_to_index[data]
        next_index = self._next_index[index]
        if next_index is None:
            return None
        return self._memory[next_index]


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
        super().__init__(
            objective=objective,
            max_trials=max_trials,
            hyperparameters=hyperparameters,
            tune_new_entries=tune_new_entries,
            allow_new_entries=allow_new_entries,
            seed=seed,
            max_retries_per_trial=max_retries_per_trial,
            max_consecutive_failed_trials=max_consecutive_failed_trials,
        )
        # List of trial_id sorting in ascending alphabetical order of their hp
        # values.
        self._ordered_ids = LinkedList()
        # Queue of trial_ids pending to find their next combinations.
        self._populate_next = []

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
        values = None

        # See if this is the first trial.
        if len(self.start_order) == 0:
            # Use all default values for the first trial.
            self._ordered_ids.insert(trial_id)
            hps = self.get_space()
            values = {
                hp.name: hp.default
                for hp in self.get_space().space
                if hps.is_active(hp)
            }
            # Although the trial is not finished, we still push it into
            # _populate_next to quickly generate values for the first few trials
            # for multiple workers. The same trial_id will be pushed into
            # _populate_next again when the trial is finished just in case of
            # new hps appeared during the trial.
            self._populate_next.append(trial_id)

        # Pick tried values to create its next combination if not tried.
        while len(self._populate_next) > 0 and values is None:
            old_trial_id = self._populate_next.pop(0)

            # Create its immediate next combination.
            old_values = self.trials[old_trial_id].hyperparameters.values
            new_values = self._get_next_combination(old_values)
            if new_values is None:
                continue

            # Skip if tried next combination.
            next_id = self._ordered_ids.next(old_trial_id)
            if next_id is not None:
                next_values = self.trials[next_id].hyperparameters.values
                if self._compare(new_values, next_values) >= 0:
                    continue

            self._ordered_ids.insert(trial_id, old_trial_id)

            values = new_values

        if values is not None:
            return {
                "status": trial_module.TrialStatus.RUNNING,
                "values": values,
            }

        # Wait for the ongoing trials to finish when the values queue is empty
        # in case of any new hp discovered.
        if len(self.ongoing_trials) > 0:
            return {"status": trial_module.TrialStatus.IDLE, "values": None}

        # Reaching this point means ongoing_trial, values, populate_next
        # are all empty.
        return {"status": trial_module.TrialStatus.STOPPED, "values": None}

    def _compare(self, a, b):
        """Compare two `HyperParameters`' values.

        The smallest index where a differs from b decides which one is larger.
        In the values of one `HyperParameter`, the default value is the smallest.
        The rest are sorted according to their order in `HyperParameter.values`.
        If one value is the prefix of another, the longer one is larger.

        Args:
            a: Dict. HyperParameters values. Only active values are included.
            b: Dict. HyperParameters values. Only active values are included.

        Returns:
            -1 if a < b, 0 if a == b, 1 if a > b.
        """
        hps = self.get_space()
        for hp in hps.space:
            # The hp is not active in neither a or b.
            # Whether it is active should be the same in a and b,
            # or the loop have stopped at the parent values which are different.
            if hp.name not in a:
                continue

            if a[hp.name] == b[hp.name]:
                continue

            # Get a ordered list of the values of the hp.
            value_list = list(hp.values)
            if hp.default in value_list:
                value_list.remove(hp.default)
            value_list.insert(0, hp.default)

            index_a = value_list.index(a[hp.name])
            index_b = value_list.index(b[hp.name])
            return -1 if index_a < index_b else 1

        return 0

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
            Dict or None. The next possible value combination for the
            hyperparameters. If no next combination exist (values is the last
            combination), it returns None. The return values only include the
            active ones.
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
        hps.values = copy.deepcopy(values)

        bumped_value = False

        # Iterate in reverse order so that we can change the value under
        # conditional scope first instead of change the condition value first.
        for hp in reversed(hps.space):
            name = hp.name
            # Bump up the hp value if possible and active.
            if hps.is_active(hp):
                value = hps.values[name]
                if value != all_values[name][-1]:
                    index = all_values[name].index(value) + 1
                    hps.values[name] = all_values[name][index]
                    bumped_value = True
                    break
            # Otherwise, reset to its first value.
            hps.values[name] = default_values[name]

        hps.ensure_active_values()
        return hps.values if bumped_value else None

    @oracle_module.synchronized
    def end_trial(self, trial):
        super().end_trial(trial)
        # It is OK for a trial_id to be pushed into _populate_next multiple
        # times. It will be skipped during _populate_space if its next
        # combination has been tried.

        # For not blocking _populate_space, we push it regardless of the status.
        self._populate_next.append(trial.trial_id)


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
        super().__init__(oracle, hypermodel, **kwargs)
