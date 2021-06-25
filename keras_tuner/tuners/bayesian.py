import math
import random

import numpy as np
import tensorflow as tf
from scipy import optimize as scipy_optimize

from ..engine import hyperparameters as hp_module
from ..engine import multi_execution_tuner
from ..engine import oracle as oracle_module
from ..engine import trial as trial_lib


def cdist(x, y=None):
    if y is None:
        y = x
    return np.linalg.norm(x[:, None, :] - y[None, :, :], axis=-1)


def solve_triangular(a, b, lower):
    if np.isfinite(a).all() and np.isfinite(b).all():
        a = tf.constant(a, dtype=tf.float32)
        b = tf.constant(b, dtype=tf.float32)
        return tf.linalg.triangular_solve(a, b, lower=lower).numpy()
    else:
        raise ValueError("array must not contain infs or NaNs")


def cho_solve(l_matrix, b):
    # Ax=b LL^T=A => Ly=b L^Tx=y
    y = solve_triangular(l_matrix, b.reshape(-1, 1), lower=True)
    return solve_triangular(l_matrix.T, y.reshape(-1, 1), lower=False)


def matern_kernel(x, y=None):
    # nu = 2.5
    dists = cdist(x, y)
    dists *= math.sqrt(5)
    kernel_matrix = (1.0 + dists + dists ** 2 / 3.0) * np.exp(-dists)
    return kernel_matrix


class GaussianProcessRegressor(object):
    """A Gaussian process regressor.

    Args:
        alpha: Float, the value added to the diagonal of the kernel matrix
            during fitting. It represents the expected amount of noise in the
            observed performances in Bayesian optimization.
        seed: Optional int, the random seed.
    """

    def __init__(self, alpha, seed=None):
        self.kernel = matern_kernel
        self.n_restarts_optimizer = 20
        self.normalize_y = True
        self.alpha = alpha
        self.seed = seed
        self._x = None
        self._y = None

    def fit(self, x, y):
        """Fit the Gaussian process regressor.

        Args:
            x: np.ndarray with shape (samples, features).
            y: np.ndarray with shape (samples,).
        """
        self._x_train = np.copy(x)
        self._y_train = np.copy(y)

        # Normalize y.
        self._y_train_mean = np.mean(self._y_train, axis=0)
        self._y_train_std = np.std(self._y_train, axis=0)
        self._y_train = (self._y_train - self._y_train_mean) / self._y_train_std

        # TODO: choose a theta for the kernel.
        kernel_matrix = self.kernel(self._x_train)
        kernel_matrix[np.diag_indices_from(kernel_matrix)] += self.alpha

        # l_matrix * l_matrix^T == kernel_matrix
        self._l_matrix = np.linalg.cholesky(kernel_matrix)
        self._alpha_vector = cho_solve(self._l_matrix, self._y_train)

    def predict(self, x):
        """Predict the mean and standard deviation of the target.

        Args:
            x: np.ndarray with shape (samples, features).

        Returns:
            Two 1-D vectors, the mean vector and standard deviation vector.
        """
        # Compute the mean.
        kernel_trans = self.kernel(x, self._x_train)
        y_mean = kernel_trans.dot(self._alpha_vector)

        # Compute the variance.
        l_inv = solve_triangular(
            self._l_matrix.T, np.eye(self._l_matrix.shape[0]), lower=False
        )
        kernel_inv = l_inv.dot(l_inv.T)

        y_var = np.ones(len(x), dtype=np.float)
        y_var -= np.einsum(
            "ij,ij->i", np.dot(kernel_trans, kernel_inv), kernel_trans
        )
        y_var[y_var < 0] = 0.0

        # Undo normalize y.
        y_var *= self._y_train_std ** 2
        y_mean = self._y_train_std * y_mean + self._y_train_mean

        return y_mean.flatten(), np.sqrt(y_var)


class BayesianOptimizationOracle(oracle_module.Oracle):
    """Bayesian optimization oracle.

    It uses Bayesian optimization with a underlying Gaussian process model.
    The acquisition function used is upper confidence bound (UCB), which can
    be found [here](
    https://www.cse.wustl.edu/~garnett/cse515t/spring_2015/files/lecture_notes/12.pdf).

    Args:
        objective: A string or `keras_tuner.Objective` instance. If a string,
            the direction of the optimization (min or max) will be inferred.
        max_trials: Integer, the total number of trials (model configurations)
            to test at most. Note that the oracle may interrupt the search
            before `max_trial` models have been tested if the search space has
            been exhausted.
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
    """

    def __init__(
        self,
        objective,
        max_trials,
        num_initial_points=None,
        alpha=1e-4,
        beta=2.6,
        seed=None,
        hyperparameters=None,
        allow_new_entries=True,
        tune_new_entries=True,
    ):
        super(BayesianOptimizationOracle, self).__init__(
            objective=objective,
            max_trials=max_trials,
            hyperparameters=hyperparameters,
            tune_new_entries=tune_new_entries,
            allow_new_entries=allow_new_entries,
            seed=seed,
        )
        self.num_initial_points = num_initial_points
        self.alpha = alpha
        self.beta = beta
        self.seed = seed or random.randint(1, int(1e4))
        self._seed_state = self.seed
        self._tried_so_far = set()
        self._max_collisions = 20
        self._random_state = np.random.RandomState(self.seed)
        self.gpr = self._make_gpr()

    def _make_gpr(self):
        return GaussianProcessRegressor(
            alpha=self.alpha,
            seed=self.seed,
        )

    def populate_space(self, trial_id):
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
        try:
            self.gpr.fit(x, y)
        except ValueError as e:
            if "array must not contain infs or NaNs" in str(e):
                return self._random_populate_space()
            raise e

        def _upper_confidence_bound(x):
            x = x.reshape(1, -1)
            mu, sigma = self.gpr.predict(x)
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
            result = scipy_optimize.minimize(
                _upper_confidence_bound, x0=x_try, bounds=bounds, method="L-BFGS-B"
            )
            if result.fun[0] < optimal_val:
                optimal_val = result.fun[0]
                optimal_x = result.x

        values = self._vector_to_values(optimal_x)
        return {"status": trial_lib.TrialStatus.RUNNING, "values": values}

    def _random_populate_space(self):
        values = self._random_values()
        if values is None:
            return {"status": trial_lib.TrialStatus.STOPPED, "values": None}
        return {"status": trial_lib.TrialStatus.RUNNING, "values": values}

    def get_state(self):
        state = super(BayesianOptimizationOracle, self).get_state()
        state.update(
            {
                "num_initial_points": self.num_initial_points,
                "alpha": self.alpha,
                "beta": self.beta,
            }
        )
        return state

    def set_state(self, state):
        super(BayesianOptimizationOracle, self).set_state(state)
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
                prob = hp_module.value_to_cumulative_prob(trial_value, hp)
                vector.append(prob)

            if trial in ongoing_trials:
                # "Hallucinate" the results of ongoing trials. This ensures that
                # repeat trials are not selected when running distributed.
                x_h = np.array(vector).reshape((1, -1))
                y_h_mean, y_h_std = self.gpr.predict(x_h)
                # Give a pessimistic estimate of the ongoing trial.
                score = y_h_mean[0] + y_h_std[0]
            elif trial.status == "COMPLETED":
                score = trial.score
                # Always frame the optimization as a minimization for scipy.minimize.
                if self.objective.direction == "max":
                    score = -1 * score
            else:
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
                value = hp_module.cumulative_prob_to_value(prob, hp)

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
        bounds = []
        for hp in self._nonfixed_space():
            bounds.append([0, 1])
        return np.array(bounds)


class BayesianOptimization(multi_execution_tuner.MultiExecutionTuner):
    """BayesianOptimization tuning with Gaussian process.

    Args:
        hypermodel: A `HyperModel` instance (or callable that takes
            hyperparameters and returns a Model instance).
        objective: A string or `keras_tuner.Objective` instance. If a string,
            the direction of the optimization (min or max) will be inferred.
        max_trials: Integer, the total number of trials (model configurations)
            to test at most. Note that the oracle may interrupt the search
            before `max_trial` models have been tested if the search space has
            been exhausted.
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
        **kwargs: Keyword arguments relevant to all `Tuner` subclasses. Please
            see the docstring for `Tuner`.
    """

    def __init__(
        self,
        hypermodel,
        objective,
        max_trials,
        num_initial_points=2,
        alpha=1e-4,
        beta=2.6,
        seed=None,
        hyperparameters=None,
        tune_new_entries=True,
        allow_new_entries=True,
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
        )
        super(
            BayesianOptimization,
            self,
        ).__init__(oracle=oracle, hypermodel=hypermodel, **kwargs)
