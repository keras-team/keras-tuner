from ..engine import tuner as tuner_module
from ..engine import oracle as oracle_module


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
