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

import numpy as np

try:
    import sklearn
    import sklearn.exceptions
    import sklearn.gaussian_process
except ImportError:  # pragma: no cover
    sklearn = None  # pragma: no cover

try:
    import scipy
    import scipy.optimize
except ImportError:  # pragma: no cover
    scipy = None  # pragma: no cover

from keras_tuner.engine import hyperparameters as hp_module
from keras_tuner.engine import oracle as oracle_module
from keras_tuner.engine import trial as trial_module
from keras_tuner.engine import tuner as tuner_module


class BayesianOptimizationOracle(oracle_module.Oracle):
    """Bayesian optimization oracle.

    It uses Bayesian optimization with a underlying Gaussian process model.
    The acquisition function used is upper confidence bound (UCB), which can
    be found [here](
    https://www.cse.wustl.edu/~garnett/cse515t/spring_2015/files/lecture_notes/12.pdf).

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
            been exhausted. Defaults to 10.
        num_initial_points: Optional number of randomly generated samples as
            initial training data for Bayesian optimization. If left
            unspecified, a value of 3 times the dimensionality of the
            hyperparameter space is used.
        alpha: Float, the value added to the diagonal of the kernel matrix
            during fitting. It represents the expected amount of noise in the
            observed performances in Bayesian optimization. Defaults to 1e-4.
        beta: Float, the balancing factor of exploration and exploitation. The
            larger it is, the more explorative it is. Defaults to 2.6.
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
        max_trials=10,
        num_initial_points=None,
        alpha=1e-4,
        beta=2.6,
        seed=None,
        hyperparameters=None,
        allow_new_entries=True,
        tune_new_entries=True,
        max_retries_per_trial=0,
        max_consecutive_failed_trials=3,
    ):
        if scipy is None:
            raise ImportError(
                "Please install scipy before using the `BayesianOptimization`."
            )

        if sklearn is None:
            raise ImportError(
                "Please install scikit-learn (sklearn) before using the "
                "`BayesianOptimization`."
            )
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
        self.num_initial_points = num_initial_points
        self.alpha = alpha
        self.beta = beta
        self._random_state = np.random.RandomState(self.seed)
        self.gpr = self._make_gpr()

    def _make_gpr(self):
        return sklearn.gaussian_process.GaussianProcessRegressor(
            kernel=sklearn.gaussian_process.kernels.Matern(nu=2.5),
            n_restarts_optimizer=20,
            normalize_y=True,
            alpha=self.alpha,
            random_state=self.seed,
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
        # Generate enough samples before training Gaussian process.
        completed_trials = [
            t for t in self.trials.values() if t.status == "COMPLETED"
        ]

        # Use 3 times the dimensionality of the space as the default number of
        # random points.
        dimensions = len(self.hyperparameters.space)
        num_initial_points = self.num_initial_points or 3 * dimensions
        if len(completed_trials) < num_initial_points:
            return self._random_populate_space()

        # Fit a GPR to the completed trials and return the predicted optimum values.
        x, y = self._vectorize_trials()

        # Ensure no nan, inf in x, y. GPR cannot process nan or inf.
        x = np.nan_to_num(x, posinf=0, neginf=0)
        y = np.nan_to_num(y, posinf=0, neginf=0)

        self.gpr.fit(x, y)

        def _upper_confidence_bound(x):
            x = x.reshape(1, -1)
            mu, sigma = self.gpr.predict(x, return_std=True)
            return mu - self.beta * sigma

        optimal_val = float("inf")
        optimal_x = None
        num_restarts = 50
        bounds = self._get_hp_bounds()
        x_seeds = self._random_state.uniform(
            bounds[:, 0], bounds[:, 1], size=(num_restarts, bounds.shape[0])
        )
        for x_try in x_seeds:
            # Sign of score is flipped when maximizing.
            result = scipy.optimize.minimize(
                _upper_confidence_bound, x0=x_try, bounds=bounds, method="L-BFGS-B"
            )
            result_fun = result.fun if np.isscalar(result.fun) else result.fun[0]
            if result_fun < optimal_val:
                optimal_val = result_fun
                optimal_x = result.x

        values = self._vector_to_values(optimal_x)
        return {"status": trial_module.TrialStatus.RUNNING, "values": values}

    def _random_populate_space(self):
        values = self._random_values()
        if values is None:
            return {"status": trial_module.TrialStatus.STOPPED, "values": None}
        return {"status": trial_module.TrialStatus.RUNNING, "values": values}

    def get_state(self):
        state = super().get_state()
        state.update(
            {
                "num_initial_points": self.num_initial_points,
                "alpha": self.alpha,
                "beta": self.beta,
            }
        )
        return state

    def set_state(self, state):
        super().set_state(state)
        self.num_initial_points = state["num_initial_points"]
        self.alpha = state["alpha"]
        self.beta = state["beta"]
        self.gpr = self._make_gpr()

    def _vectorize_trials(self):
        x = []
        y = []
        ongoing_trials = set(self.ongoing_trials.values())
        for trial in self.trials.values():
            # Create a vector representation of each Trial's hyperparameters.
            trial_hps = trial.hyperparameters
            vector = []
            for hp in self._nonfixed_space():
                # For hyperparameters not present in the trial (either added after
                # the trial or inactive in the trial), set to default value.
                if (
                    trial_hps.is_active(hp)  # inactive
                    and hp.name in trial_hps.values  # added after the trial
                ):
                    trial_value = trial_hps.values[hp.name]
                else:
                    trial_value = hp.default

                # Embed an HP value into the continuous space [0, 1].
                prob = hp.value_to_prob(trial_value)
                vector.append(prob)

            if trial in ongoing_trials:
                # "Hallucinate" the results of ongoing trials. This ensures that
                # repeat trials are not selected when running distributed.
                x_h = np.array(vector).reshape((1, -1))
                y_h_mean, y_h_std = self.gpr.predict(x_h, return_std=True)
                # Give a pessimistic estimate of the ongoing trial.
                score = y_h_mean[0] + y_h_std[0]
            elif trial.status == "COMPLETED":
                score = trial.score
                # Always frame the optimization as a minimization for scipy.minimize.
                if self.objective.direction == "max":
                    score = -1 * score
            else:
                # Skip the failed and invalid trials.
                continue

            x.append(vector)
            y.append(score)

        x = np.array(x)
        y = np.array(y)
        return x, y

    def _vector_to_values(self, vector):
        hps = hp_module.HyperParameters()
        vector_index = 0
        for hp in self.hyperparameters.space:
            hps.merge([hp])
            if isinstance(hp, hp_module.Fixed):
                value = hp.value
            else:
                prob = vector[vector_index]
                vector_index += 1
                value = hp.prob_to_value(prob)

            if hps.is_active(hp):
                hps.values[hp.name] = value
        return hps.values

    def _find_closest(self, val, hp):
        values = [hp.min_value]
        while values[-1] + hp.step <= hp.max_value:
            values.append(values[-1] + hp.step)

        array = np.asarray(values)
        index = (np.abs(values - val)).argmin()
        return array[index]

    def _nonfixed_space(self):
        return [
            hp
            for hp in self.hyperparameters.space
            if not isinstance(hp, hp_module.Fixed)
        ]

    def _get_hp_bounds(self):
        bounds = [[0, 1] for _ in self._nonfixed_space()]
        return np.array(bounds)


class BayesianOptimization(tuner_module.Tuner):
    """BayesianOptimization tuning with Gaussian process.

    Args:
        hypermodel: Instance of `HyperModel` class (or callable that takes
            hyperparameters and returns a `Model` instance). It is optional
            when `Tuner.run_trial()` is overriden and does not use
            `self.hypermodel`.
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
            been exhausted. Defaults to 10.
        num_initial_points: Optional number of randomly generated samples as
            initial training data for Bayesian optimization. If left
            unspecified, a value of 3 times the dimensionality of the
            hyperparameter space is used.
        alpha: Float, the value added to the diagonal of the kernel matrix
            during fitting. It represents the expected amount of noise in the
            observed performances in Bayesian optimization. Defaults to 1e-4.
        beta: Float, the balancing factor of exploration and exploitation. The
            larger it is, the more explorative it is. Defaults to 2.6.
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
        **kwargs: Keyword arguments relevant to all `Tuner` subclasses. Please
            see the docstring for `Tuner`.
    """

    def __init__(
        self,
        hypermodel=None,
        objective=None,
        max_trials=10,
        num_initial_points=None,
        alpha=1e-4,
        beta=2.6,
        seed=None,
        hyperparameters=None,
        tune_new_entries=True,
        allow_new_entries=True,
        max_retries_per_trial=0,
        max_consecutive_failed_trials=3,
        **kwargs
    ):
        oracle = BayesianOptimizationOracle(
            objective=objective,
            max_trials=max_trials,
            num_initial_points=num_initial_points,
            alpha=alpha,
            beta=beta,
            seed=seed,
            hyperparameters=hyperparameters,
            tune_new_entries=tune_new_entries,
            allow_new_entries=allow_new_entries,
            max_retries_per_trial=max_retries_per_trial,
            max_consecutive_failed_trials=max_consecutive_failed_trials,
        )
        super().__init__(oracle=oracle, hypermodel=hypermodel, **kwargs)
