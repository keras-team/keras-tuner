import numpy as np
import random
from scipy import optimize as scipy_optimize
from scipy import stats as scipy_stats
from sklearn import gaussian_process

from ..engine import tuner as tuner_module
from ..engine import oracle as oracle_module
from ..engine import hyperparameters as hp_module


class BayesianOptimizationOracle(oracle_module.Oracle):
    def __init__(self,
                 init_samples=2,
                 alpha=0.04,
                 beta=0.01,
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

        x, y = self._training_data()
        # Update Gaussian process with existing samples
        self.gpr.fit(x, y)

        values = self.propose_location(x.shape[1])
        return self._convert(values)

    def report_status(self, trial_id, status):
        # TODO
        raise NotImplementedError

    def save(self):
        # TODO
        raise NotImplementedError

    @classmethod
    def load(cls, filename):
        # TODO
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

        return np.array(x), np.array(y)

    def _convert(self, vector):
        values = {}
        for index, value in enumerate(vector):
            hp = self.space[index]
            if isinstance(hp, hp_module.Choice):
                value = hp.values[value]
            elif isinstance(hp, hp_module.Fixed):
                value = hp.value
            values[hp.name] = value
        return values

    def _get_hp_index(self, name):
        for index, hp in enumerate(self.space):
            if hp.name == name:
                return index
        return None

    def expected_improvement(self, X, X_sample):
        """Computes the EI at points X based on existing samples X_sample and
        Y_sample using a Gaussian process surrogate model.

        Args:
            X: Points at which EI shall be computed (m x d).
            X_sample: Sample locations (n x d).

        Returns:
            Expected improvements at points X.
        """
        mu, sigma = self.gpr.predict(X, return_std=True)
        mu_sample = self.gpr.predict(X_sample)

        sigma = sigma.reshape(-1, X_sample.shape[1])

        # Needed for noise-based model,
        # otherwise use np.max(Y_sample).
        # See also section 2.4 in [...]
        mu_sample_opt = np.max(mu_sample)

        with np.errstate(divide='warn'):
            imp = mu - mu_sample_opt - self.beta
            Z = imp / sigma
            ei = imp * scipy_stats.norm.cdf(Z) + sigma * scipy_stats.norm.pdf(Z)
            ei[sigma == 0.0] = 0.0

        return ei

    def propose_location(self, dim):
        """ Proposes the next sampling point by optimizing the acquisition function.

        Args:
            dim: Int. x.shape[1]
        Returns:
            Location of the acquisition function maximum.
        """
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

        def min_obj(x):
            # Minimization objective is the negative acquisition function
            return -self.expected_improvement(x.reshape(-1, dim), x)

        # Find the best optimum by starting from n_restart different random points.
        n_restarts = 25
        for x0 in np.random.uniform(bounds[:, 0], bounds[:, 1],
                                    size=(n_restarts, dim)):
            res = scipy_optimize.minimize(min_obj, x0=x0, bounds=bounds,
                                          method='L-BFGS-B')
            if res.fun < min_val:
                min_val = res.fun[0]
                min_x = res.x

        return min_x.reshape(-1, 1)


class BayesianOptimization(tuner_module.Tuner):
    def __init__(self,
                 hypermodel,
                 objective,
                 max_trials,
                 init_samples=2,
                 seed=None,
                 **kwargs):
        oracle = BayesianOptimizationOracle(init_samples, seed)
        super(BayesianOptimization, self, ).__init__(oracle=oracle,
                                                     hypermodel=hypermodel,
                                                     objective=objective,
                                                     max_trials=max_trials,
                                                     **kwargs)
