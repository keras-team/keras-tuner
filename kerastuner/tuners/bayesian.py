import numpy as np
import random
import json
from scipy import optimize as scipy_optimize
from sklearn import gaussian_process

from ..abstractions.tensorflow import TENSORFLOW_UTILS as tf_utils
from ..engine import tuner as tuner_module
from ..engine import oracle as oracle_module
from ..engine import hyperparameters as hp_module


class BayesianOptimizationOracle(oracle_module.Oracle):
    """Bayesian optimization oracle.

    It uses Bayesian optimization with a underlying Gaussian process model.
    The acquisition function used is upper confidence bound (UCB).

    Attributes:
        init_samples: Int. The number of randomly generated samples as initial
            training data for Bayesian optmization.
        alpha: Float or array-like. Value added to the diagonal of
            the kernel matrix during fitting.
        beta: Float. The balancing factor of exploration and exploitation.
            The larger it is, the more explorative it is.
        seed: Int. Random seed.

    """

    def __init__(self,
                 init_samples=2,
                 alpha=1e-10,
                 beta=2.6,
                 seed=None):
        super(BayesianOptimizationOracle, self).__init__()
        self.init_samples = init_samples
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

    def populate_space(self, trial_id, space):
        self.update_space(space)
        # Generate enough samples before training Gaussian process.
        if self._num_trials < self.init_samples or len(self._score) < 2:
            self._num_trials += 1
            values = self._new_trial()
            self._values[trial_id] = values
            return {'status': 'RUN', 'values': values}

        self._training_data()
        # Update Gaussian process with existing samples
        self.gpr.fit(self._x, self._y)

        values = self._generate_vector()
        return {'status': 'RUN', 'values': self._convert(values)}

    def report_status(self, trial_id, status):
        # TODO
        raise NotImplementedError

    def save(self, fname):
        state = {
            'init_samples': self.init_samples,
            'alpha': self.alpha,
            'beta': self.beta,
            'seed': self.seed,
            'seed_state': self._seed_state,
            'tried_so_far': list(self._tried_so_far),
            'max_collisions': self._max_collisions,
            'num_trials': self._num_trials,
            'score': self._score,
            'values': self._values,
            'x': self._x,
            'y': self._y,
        }
        state_json = json.dumps(state)
        tf_utils.write_file(fname, state_json)

    def reload(self, fname):
        state_data = tf_utils.read_file(fname)
        state = json.loads(state_data)
        self.init_samples = state['init_samples']
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
        self.gpr = gaussian_process.GaussianProcessRegressor(
            kernel=gaussian_process.kernels.ConstantKernel(1.0),
            alpha=self.alpha)
        raise NotImplementedError

    def result(self, trial_id, score):
        self._score[trial_id] = score

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
            for p in self.space:
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

    def _training_data(self):
        x = []
        y = []
        for trial_id in self._values:
            values = self._values[trial_id]
            score = self._score[trial_id]

            vector = [None] * len(values)
            for name, value in values.items():
                index = self._get_hp_index(name)
                hp = self.space[index]
                if isinstance(hp, hp_module.Choice):
                    value = hp.values.index(value)
                elif isinstance(hp, hp_module.Fixed):
                    value = 0
                vector[index] = value

            x.append(vector)
            y.append(score)

        self._x = np.array(x)
        self._y = np.array(y)

    def _convert(self, vector):
        values = {}
        for index, value in enumerate(vector):
            hp = self.space[index]
            if isinstance(hp, hp_module.Choice):
                value = hp.values[int(value)]
            elif isinstance(hp, hp_module.Fixed):
                value = hp.value
            values[hp.name] = value
        return values

    def _get_hp_index(self, name):
        for index, hp in enumerate(self.space):
            if hp.name == name:
                return index
        return None

    def _upper_confidence_bound(self, x):
        dim = len(self.space)
        x = x.reshape(-1, dim)
        mu, sigma = self.gpr.predict(x, return_std=True)
        return mu - self.beta * sigma

    def _generate_vector(self, ):
        dim = len(self.space)
        min_val = 1
        min_x = None
        bounds = []
        for hp in self.space:
            if isinstance(hp, hp_module.Choice):
                bound = [0, len(hp.values)]
            elif isinstance(hp, hp_module.Fixed):
                bound = [0, 0]
            else:
                bound = [hp.min_value, hp.max_value]
            bounds.append(bound)
        bounds = np.array(bounds)

        # Find the best optimum by starting from n_restart different random points.
        n_restarts = 25
        for x0 in np.random.uniform(bounds[:, 0], bounds[:, 1],
                                    size=(n_restarts, dim)):
            res = scipy_optimize.minimize(self._upper_confidence_bound,
                                          x0=x0, bounds=bounds,
                                          method='L-BFGS-B')
            if res.fun < min_val:
                min_val = res.fun[0]
                min_x = res.x

        return min_x.reshape(-1, 1)


class BayesianOptimization(tuner_module.Tuner):
    """BayesianOptimization tuning with Gaussian process.

    Args:
        hypermodel: Instance of HyperModel class
            (or callable that takes hyperparameters
            and returns a Model isntance).
        objective: String. Name of model metric to minimize
            or maximize, e.g. "val_accuracy".
        max_trials: Int. Total number of trials
            (model configurations) to test at most.
            Note that the oracle may interrupt the search
            before `max_trial` models have been tested.
        init_samples: Int. The number of randomly generated samples as initial
            training data for Bayesian optmization.
        alpha: Float or array-like. Value added to the diagonal of
            the kernel matrix during fitting.
        beta: Float. The balancing factor of exploration and exploitation.
            The larger it is, the more explorative it is.
        seed: Int. Random seed.
    """

    def __init__(self,
                 hypermodel,
                 objective,
                 max_trials,
                 init_samples=2,
                 seed=None,
                 **kwargs):
        oracle = BayesianOptimizationOracle(init_samples=init_samples,

                                            seed=seed)
        super(BayesianOptimization, self, ).__init__(oracle=oracle,
                                                     hypermodel=hypermodel,
                                                     objective=objective,
                                                     max_trials=max_trials,
                                                     **kwargs)
