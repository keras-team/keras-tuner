import numpy as np
from scipy import optimize as scipy_optimize
from scipy import stats as scipy_stats

from ..engine import tuner as tuner_module
from ..engine import oracle as oracle_module


def expected_improvement(X, X_sample, Y_sample, gpr, xi=0.01):
    """ Computes the EI at points X based on existing samples X_sample and
    Y_sample using a Gaussian process surrogate model.
    Args:
        X: Points at which EI shall be computed (m x d).
        X_sample: Sample locations (n x d).
        Y_sample: Sample values (n x 1).
        gpr: A GaussianProcessRegressor fitted to samples.
        xi: Exploitation-exploration trade-off parameter.
    Returns:
        Expected improvements at points X.
    """
    mu, sigma = gpr.predict(X, return_std=True)
    mu_sample = gpr.predict(X_sample)

    sigma = sigma.reshape(-1, X_sample.shape[1])

    # Needed for noise-based model,
    # otherwise use np.max(Y_sample).
    # See also section 2.4 in [...]
    mu_sample_opt = np.max(mu_sample)

    with np.errstate(divide='warn'):
        imp = mu - mu_sample_opt - xi
        Z = imp / sigma
        ei = imp * scipy_stats.norm.cdf(Z) + sigma * scipy_stats.norm.pdf(Z)
        ei[sigma == 0.0] = 0.0

    return ei


def propose_location(acquisition, X_sample, Y_sample, gpr, bounds, n_restarts=25):
    """ Proposes the next sampling point by optimizing the acquisition function.

    Args:
        acquisition: Acquisition function. X_sample: Sample locations (n x d).
        Y_sample: Sample values (n x 1). gpr: A GaussianProcessRegressor fitted to
            samples.
    Returns:
        Location of the acquisition function maximum.
    """
    dim = X_sample.shape[1]
    min_val = 1
    min_x = None

    def min_obj(X):
        # Minimization objective is the negative acquisition function
        return -acquisition(X.reshape(-1, dim), X_sample, Y_sample, gpr)

    # Find the best optimum by starting from n_restart different random points.
    for x0 in np.random.uniform(bounds[:, 0], bounds[:, 1], size=(n_restarts, dim)):
        res = scipy_optimize.minimize(min_obj, x0=x0, bounds=bounds, method='L-BFGS-B')
        if res.fun < min_val:
            min_val = res.fun[0]
            min_x = res.x

    return min_x.reshape(-1, 1)


from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern

# Gaussian process with Mat??rn kernel as surrogate model
m52 = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5)
gpr = GaussianProcessRegressor(kernel=m52, alpha=noise ** 2)

# Initialize samples
X_sample = X_init
Y_sample = Y_init

# Number of iterations
n_iter = 10

for i in range(n_iter):
    # Update Gaussian process with existing samples
    gpr.fit(X_sample, Y_sample)

    # Obtain next sampling point from the acquisition function (expected_improvement)
    X_next = propose_location(expected_improvement, X_sample, Y_sample, gpr, bounds)

    # Obtain next noisy sample from the objective function
    Y_next = f(X_next, noise)

    # Add sample to previous samples
    X_sample = np.vstack((X_sample, X_next))
    Y_sample = np.vstack((Y_sample, Y_next))


class BayesianOptimizationOracle(oracle_module.Oracle):
    def populate_space(self, trial_id, space):
        pass

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


class BayesianOptimization(tuner_module.Tuner):
    pass
