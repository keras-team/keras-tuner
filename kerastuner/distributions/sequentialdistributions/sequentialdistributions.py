from kerastuner.distributions.distributions import Distributions
from kerastuner.abstractions.display import fatal
import random
from numpy import linspace, logspace
from collections import defaultdict


class SequentialDistributions(Distributions):
    """Random distributions

    Args:
        hyperparameters_config (dict): hyperparameters dict describing
        the search space. Often refered as hparams. Generated using
        DummyDistributions() in Tuner()
    Attributes:
        counters (defaultdict(list)): track what is the current returned value
    """

    def __init__(self, hyperparameters_config):
        super(SequentialDistributions, self).__init__('SequentialDistributions',  # nopep8
                                                      hyperparameters_config)
        self.counters = defaultdict(int)
        self.ranges = {}

    def Fixed(self, name, value, group="default"):
        """Return a fixed selected value
        Args:
            name (str): name of the parameter
            value: value of the parameter
            group (str): Optional logical grouping of the parameters
        Returns:
            fixed value
        """
        self._record_hyperparameter(name, value, group)
        return value

    def Boolean(self, name, group="default"):
        """Return a random Boolean value.
        Args:
            name (str): name of the parameter
            group (str): Optional logical grouping of the parameters
        Returns:
            an Boolean
        """
        key = self._get_key(name, group)
        if key not in self.ranges:
            self.ranges[key] = [True, False]

        value = self._get_next_value(key)
        self._record_hyperparameter(name, value, group)
        return value

    def Choice(self, name, selection, group="default"):
        """Return a random value from an explicit list of choice.
        Args:
            name (str): name of the parameter
            selection (list): list of explicit choices
            group (str): Optional logical group name this parameter belongs to
        Returns:
            an element of the list provided
        """
        key = self._get_key(name, group)
        if key not in self.ranges:
            self.ranges[key] = selection

        value = self._get_next_value(key)
        if isinstance(selection[0], int):
            value = int(value)
        elif isinstance(selection[0], float):
            value = float(value)
        elif isinstance(selection[0], str):
            value = str(value)
        else:
            Exception('unknown type')
        self._record_hyperparameter(name, value, group)
        return value

    def Range(self, name, start, stop, increment=1, group='default'):
        """Return a random value from a range.
        Args:
            name (str): name of the parameter
            start (int/float): lower bound of the range
            stop (int/float): upper bound of the range
            increment (int/float): incremental step
        Returns:
            an element of the range
        """
        key = self._get_key(name, group)
        if key not in self.ranges:
            self.ranges[key] = list(range(start, stop, increment))

        value = self._get_next_value(key)
        self._record_hyperparameter(name, value, group)
        return value

    def Logarithmic(self, name, start, stop, num_buckets, precision=0,
                    group='default'):
        """Return a random value from a range which is logarithmically divided.
        Args:
            name (str): name of the parameter
            start (int/float): lower bound of the range
            stop (int/float): upper bound of the range
            num_buckets (int): into how many buckets to divided the range in
            precision (int): For float range. Round the result rounded to the
                            nth decimal if needed. 0 means not rounded
        Returns:
            an element of the range
        """
        key = self._get_key(name, group)
        if key not in self.ranges:
            self.ranges[key] = logspace(start, stop, num_buckets)

        value = self._get_next_value(key)
        self._record_hyperparameter(name, value, group)
        return value

    def Linear(self, name, start, stop, num_buckets, precision=0,
               group='default'):
        """Return a random value from a range which is linearly divided.
        Args:
            name (str): name of the parameter
            start (int/float): lower bound of the range
            stop (int/float): upper bound of the range
            num_buckets (int): into how many buckets to divided the range in
            precision (int): For float range. Round the result rounded to the
                            nth decimal if needed. 0 means not rounded
        Returns:
            an element of the range
        """
        key = self._get_key(name, group)
        if key not in self.ranges:
            self.ranges[name] = linspace(start, stop, num_buckets)

        value = self._get_next_value(key)
        if precision:
            value = round(value, precision)

        self._record_hyperparameter(name, value, group)
        return value

    def _get_next_value(self, key):
        "Return next value of the range"
        # counter index with wrap around

        idx = self.counters[key]

        idx = idx % len(self.ranges[key])
        value = self.ranges[key][idx]

        # increment counter will wrap-around if needed next time
        self.counters[key] = idx + 1

        return value
