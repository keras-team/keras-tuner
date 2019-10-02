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
        self._num_trials = 0
        self._score = {}
        self._values = {}
        self._x = None
        self._y = None
        self.gpr = gaussian_process.GaussianProcessRegressor(
            kernel=gaussian_process.kernels.ConstantKernel(1.0),
            alpha=self.alpha)

    def _populate_space(self, trial_id):
        # Generate enough samples before training Gaussian process.
        if self._num_trials < self.num_initial_points or len(self._score) < 2:
            self._num_trials += 1
            values = self._new_trial()
            self._values[trial_id] = values
            return {'status': trial_lib.TrialStatus.RUNNING,
                    'values': values}
        values = self._to_hp_dict(self._generate_vector())
        self._values[trial_id] = values
        return {'status': trial_lib.TrialStatus.RUNNING,
                'values': values}

    def end_trial(self, trial_id, status):
        super(BayesianOptimizationOracle, self).end_trial(trial_id, status)
        if status == trial_lib.TrialStatus.COMPLETED:
            self._score[trial_id] = self.trials[trial_id].score
            # Update Gaussian process with existing samples
            if len(self._score) >= self.num_initial_points:
                x, y = self._get_training_data()
                self.gpr.fit(x, y)
                self._x = x
                self._y = y

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
            'num_trials': self._num_trials,
            'score': self._score,
            'values': self._values,
            'x': self._x.tolist() if isinstance(self._x, np.ndarray) else self._x,
            'y': self._y.tolist() if isinstance(self._y, np.ndarray) else self._y,
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
        self._num_trials = state['num_trials']
        self._score = state['score']
        self._values = state['values']
        self._x = state['x']
        self._y = state['y']
        # Remove the unfinished trial_id.
        for key in [key for key in self._values if key not in self._score]:
            self._tried_so_far.remove(self._compute_values_hash(self._values[key]))
            self._values.pop(key)
        self.gpr = gaussian_process.GaussianProcessRegressor(
            kernel=gaussian_process.kernels.ConstantKernel(1.0),
            alpha=self.alpha)

    def _new_trial(self):
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

    def _get_training_data(self):
        x = []
        y = []
        for trial_id in self._values:
            values = self._values[trial_id]
            if trial_id not in self._score:
                continue
            score = self._score[trial_id]

            vector = [0] * len(self.hyperparameters.space)
            for name, value in values.items():
                index = self._get_hp_index(name)
                hp = self.hyperparameters.space[index]
                if isinstance(hp, hp_module.Choice):
                    value = hp.values.index(value)
                elif isinstance(hp, hp_module.Fixed):
                    value = 0
                elif isinstance(hp, hp_module.Int):
                    value = value
                elif isinstance(hp, hp_module.Float):
                    value = value
                vector[index] = value

            # Exclude the fixed value from the vector
            no_fixed_vector = []
            for index, value in enumerate(vector):
                if not isinstance(self.hyperparameters.space[index],
                                  hp_module.Fixed):
                    no_fixed_vector.append(value)
            x.append(no_fixed_vector)
            y.append(score)
        x = np.array(x)
        max_score = np.max(np.abs(y))
        if max_score > 0:
            y = np.array(y) / max_score
        return x, y

    def _to_hp_dict(self, vector):
        values = {}
        vector_index = 0
        for index, hp in enumerate(self.hyperparameters.space):
            hp = self.hyperparameters.space[index]
            if isinstance(hp, hp_module.Fixed):
                values[hp.name] = hp.value
                continue

            value = vector[vector_index]
            vector_index += 1

            if isinstance(hp, hp_module.Choice):
                value = hp.values[int(value)]
            elif isinstance(hp, hp_module.Float):
                value = value[0]
            elif isinstance(hp, hp_module.Int):
                value = value[0]
            values[hp.name] = value

        return values

    def _get_hp_index(self, name):
        for index, hp in enumerate(self.hyperparameters.space):
            if hp.name == name:
                return index
        return None

    def _upper_confidence_bound(self, x):
        x = x.reshape(-1, self._dim)
        mu, sigma = self.gpr.predict(x, return_std=True)
        return mu - self.beta * sigma

    @property
    def _dim(self):
        return len([hp for hp in self.hyperparameters.space
                    if not isinstance(hp, hp_module.Fixed)])

    def _generate_vector(self):
        min_val = float('inf')
        min_x = None
        bounds = []
        for hp in self.hyperparameters.space:
            if isinstance(hp, hp_module.Choice):
                bound = [0, len(hp.values)]
            elif isinstance(hp, hp_module.Fixed):
                # Fixed values are excluded form the vector.
                continue
            else:
                bound = [hp.min_value, hp.max_value]
            bounds.append(bound)
        bounds = np.array(bounds)

        # Find the best optimum by starting from n_restart different random points.
        n_restarts = 25
        for x in np.random.uniform(bounds[:, 0], bounds[:, 1],
                                   size=(n_restarts, self._dim)):
            res = scipy_optimize.minimize(self._upper_confidence_bound,
                                          x0=x, bounds=bounds,
                                          method='L-BFGS-B')
            if res.fun < min_val:
                min_val = res.fun[0]
                min_x = res.x

        return min_x.reshape(-1, 1)


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
