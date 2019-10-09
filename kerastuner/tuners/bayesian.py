import random

import numpy as np
from scipy import optimize as scipy_optimize
from sklearn import gaussian_process

from ..engine import hyperparameters as hp_module
from ..engine import multi_execution_tuner
from ..engine import oracle as oracle_module
from ..engine import trial as trial_lib


class BayesianOptimizationOracle(oracle_module.Oracle):
    """Bayesian optimization oracle.

    It uses Bayesian optimization with a underlying Gaussian process model.
    The acquisition function used is upper confidence bound (UCB), which can
    be found in the following link:
    https://www.cse.wustl.edu/~garnett/cse515t/spring_2015/files/lecture_notes/12.pdf

    Attributes:
        objective: String or `kerastuner.Objective`. If a string,
          the direction of the optimization (min or max) will be
          inferred.
        max_trials: Int. Total number of trials
            (model configurations) to test at most.
            Note that the oracle may interrupt the search
            before `max_trial` models have been tested.
        num_initial_points: Int. The number of randomly generated samples as initial
            training data for Bayesian optimization.
        alpha: Float. Value added to the diagonal of the kernel matrix
            during fitting. It represents the expected amount of noise
            in the observed performances in Bayesian optimization.
        beta: Float. The balancing factor of exploration and exploitation.
            The larger it is, the more explorative it is.
        seed: Int. Random seed.
        hyperparameters: HyperParameters class instance.
            Can be used to override (or register in advance)
            hyperparamters in the search space.
        tune_new_entries: Whether hyperparameter entries
            that are requested by the hypermodel
            but that were not specified in `hyperparameters`
            should be added to the search space, or not.
            If not, then the default value for these parameters
            will be used.
        allow_new_entries: Whether the hypermodel is allowed
            to request hyperparameter entries not listed in
            `hyperparameters`.
    """

    def __init__(self,
                 objective,
                 max_trials,
                 num_initial_points=2,
                 alpha=1e-4,
                 beta=2.6,
                 seed=None,
                 hyperparameters=None,
                 allow_new_entries=True,
                 tune_new_entries=True):
        super(BayesianOptimizationOracle, self).__init__(
            objective=objective,
            max_trials=max_trials,
            hyperparameters=hyperparameters,
            tune_new_entries=tune_new_entries,
            allow_new_entries=allow_new_entries)
        self.num_initial_points = num_initial_points
        self.alpha = alpha
        self.beta = beta
        self.seed = seed or random.randint(1, 1e4)
        self._seed_state = self.seed
        self._tried_so_far = set()
        self._max_collisions = 20
        # self._num_trials = 0
        # self._score = {}
        # self._values = {}
        self.gpr = gaussian_process.GaussianProcessRegressor(
            alpha=self.alpha)

    def _populate_space(self, trial_id):
        # Generate enough samples before training Gaussian process.
        completed_trials = [t for t in self.trials.values()
                            if t.status == "COMPLETED"]
        if len(self.trials) < self.num_initial_points or len(completed_trials) < 2:
            values = self._random_trial()
            return {'status': trial_lib.TrialStatus.RUNNING,
                    'values': values}

        # Fit a GPR to the completed trials and return the predicted optimum values.
        x, y = self._vectorize_trials()
        self.gpr.fit(x, y)

        def _upper_confidence_bound(x):
            x = x.reshape(1, -1)
            mu, sigma = self.gpr.predict(x, return_std=True)
            return mu - self.beta * sigma

        optimal_val = float('inf')
        optimal_x = None
        num_restarts = 25
        bounds = self._get_hp_bounds()
        for _ in range(num_restarts):
            x0 = np.random.uniform(bounds[:, 0], bounds[:, 1])
            # Sign of score is flipped when maximizing.
            result = scipy_optimize.minimize(_upper_confidence_bound,
                                             x0=x0,
                                             bounds=bounds,
                                             method='L-BFGS-B')
            if result.fun[0] < optimal_val:
                optimal_val = result.fun[0]
                optimal_x = result.x

        values = self._vector_to_values(optimal_x)
        return {'status': trial_lib.TrialStatus.RUNNING,
                'values': values}

    def get_state(self):
        state = super(BayesianOptimizationOracle, self).get_state()
        state.update({
            'num_initial_points': self.num_initial_points,
            'alpha': self.alpha,
            'beta': self.beta,
            'seed': self.seed,
            'seed_state': self._seed_state,
            'tried_so_far': list(self._tried_so_far),
            'max_collisions': self._max_collisions,
        })
        return state

    def set_state(self, state):
        super(BayesianOptimizationOracle, self).set_state(state)
        self.num_initial_points = state['num_initial_points']
        self.alpha = state['alpha']
        self.beta = state['beta']
        self.seed = state['seed']
        self._seed_state = state['seed_state']
        self._tried_so_far = set(state['tried_so_far'])
        self._max_collisions = state['max_collisions']
        self.gpr = gaussian_process.GaussianProcessRegressor(
            kernel=gaussian_process.kernels.ConstantKernel(1.0),
            alpha=self.alpha)

    def _random_trial(self):
        """Fill a given hyperparameter space with values.

        Returns:
            A dictionary mapping parameter names to suggested values.
            Note that if the Oracle is keeping tracking of a large
            space, it may return values for more parameters
            than what was listed in `space`.
        """
        collisions = 0
        while 1:
            # Generate a set of random values.
            values = {}
            for p in self.hyperparameters.space:
                values[p.name] = p.random_sample(self._seed_state)
                self._seed_state += 1
            # Keep trying until the set of values is unique,
            # or until we exit due to too many collisions.
            values_hash = self._compute_values_hash(values)
            if values_hash in self._tried_so_far:
                collisions += 1
                if collisions > self._max_collisions:
                    return None
                continue
            self._tried_so_far.add(values_hash)
            break
        return values

    def _vectorize_trials(self):
        x = []
        y = []
        for trial in self.trials.values():
            if trial.status != "COMPLETED":
                continue

            trial_values = trial.hyperparameters.values
            score = trial.score

            # Always frame the optimization as a minimization for scipy.minimize.
            if self.objective.direction == 'max':
                score = -1*score

            # Create a vector representation for each Trial's hyperparameters.
            vector = []
            for hp in self._nonfixed_space():
                # Hyperparameters could have been added to the study since
                # the trial was run.
                if hp.name in trial_values:
                    trial_value = trial_values[hp.name]
                else:
                    trial_value = hp.default

                # Embed an HP value into a continuous space.
                if isinstance(hp, hp_module.Boolean):
                    val = 1 if trial_value else 0
                elif isinstance(hp, hp_module.Choice):
                    val = hp.values.index(trial_value)
                else:
                    val = trial_value
                vector.append(val)

            x.append(vector)
            y.append(score)

        x = np.array(x)
        max_score = np.max(np.abs(y))
        if max_score > 0:
            y = np.array(y) / max_score
        return x, y

    def _vector_to_values(self, vector):
        values = {}
        vector_index = 0
        for index, hp in enumerate(self.hyperparameters.space):
            hp = self.hyperparameters.space[index]
            if isinstance(hp, hp_module.Fixed):
                values[hp.name] = hp.value
                continue

            value = vector[vector_index]
            vector_index += 1

            if isinstance(hp, hp_module.Boolean):
                value = True if int(round(value)) == 1 else False
            if isinstance(hp, hp_module.Choice):
                value = hp.values[int(round(value))]
            elif isinstance(hp, hp_module.Int):
                if hp.step is not None:
                    value = int(self._find_closest(value, hp))
                else:
                    value = round(value)
            elif isinstance(hp, hp_module.Float):
                if hp.step is not None:
                    value = self._find_closest(value, hp)
                else:
                    value = value
            values[hp.name] = value

        return values

    def _find_closest(self, val, hp):
        values = [hp.min_value]
        while values[-1] + hp.step <= hp.max_value:
            values.append(values[-1] + hp.step)

        array = np.asarray(values)
        index = (np.abs(values - val)).argmin()
        return array[index]

    def _get_hp_index(self, name):
        for index, hp in enumerate(self.hyperparameters.space):
            if hp.name == name:
                return index
        return None

    def _nonfixed_space(self):
        return [hp for hp in self.hyperparameters.space
                if not isinstance(hp, hp_module.Fixed)]

    def _get_hp_bounds(self):
        bounds = []
        for hp in self._nonfixed_space():
            if isinstance(hp, hp_module.Boolean):
                bound = [0, 1]
            elif isinstance(hp, hp_module.Choice):
                bound = [0, len(hp.values) - 1]
            else:
                bound = [hp.min_value, hp.max_value]
            bounds.append(bound)
        return np.array(bounds)


class BayesianOptimization(multi_execution_tuner.MultiExecutionTuner):
    """BayesianOptimization tuning with Gaussian process.

    Args:
        hypermodel: Instance of HyperModel class
            (or callable that takes hyperparameters
            and returns a Model instance).
        objective: String. Name of model metric to minimize
            or maximize, e.g. "val_accuracy".
        max_trials: Int. Total number of trials
            (model configurations) to test at most.
            Note that the oracle may interrupt the search
            before `max_trial` models have been tested.
        num_initial_points: Int. The number of randomly generated samples as initial
            training data for Bayesian optimization.
        alpha: Float or array-like. Value added to the diagonal of
            the kernel matrix during fitting.
        beta: Float. The balancing factor of exploration and exploitation.
            The larger it is, the more explorative it is.
        seed: Int. Random seed.
        hyperparameters: HyperParameters class instance.
            Can be used to override (or register in advance)
            hyperparamters in the search space.
        tune_new_entries: Whether hyperparameter entries
            that are requested by the hypermodel
            but that were not specified in `hyperparameters`
            should be added to the search space, or not.
            If not, then the default value for these parameters
            will be used.
        allow_new_entries: Whether the hypermodel is allowed
            to request hyperparameter entries not listed in
            `hyperparameters`.
        **kwargs: Keyword arguments relevant to all `Tuner` subclasses.
            Please see the docstring for `Tuner`.
    """

    def __init__(self,
                 hypermodel,
                 objective,
                 max_trials,
                 num_initial_points=2,
                 seed=None,
                 hyperparameters=None,
                 tune_new_entries=True,
                 allow_new_entries=True,
                 **kwargs):
        oracle = BayesianOptimizationOracle(
            objective=objective,
            max_trials=max_trials,
            num_initial_points=num_initial_points,
            seed=seed,
            hyperparameters=hyperparameters,
            tune_new_entries=tune_new_entries,
            allow_new_entries=allow_new_entries)
        super(BayesianOptimization, self, ).__init__(oracle=oracle,
                                                     hypermodel=hypermodel,
                                                     **kwargs)
