from kerastuner.engine.distributions import Distributions
from kerastuner.abstractions.display import fatal
import random
from numpy import linspace, logspace


class RandomDistributions(Distributions):
    "Random distributions"

    def __init__(self):
        self.hyperparameters_config = {}
        super(RandomDistributions, self).__init__('RandomDistributions')

    def Fixed(self, name, value, group="default"):
        """Return a fixed selected value
        Args:
            name (str): name of the parameter
            value: value of the parameter
            group (str): Optional logical grouping of the parameters
        Returns:
            fixed value
        """
        self._record_current_hyperparameters(name, value, group)
        return value

    def Boolean(self, name, group="default"):
        """Return a random Boolean value.
        Args:
            name (str): name of the parameter
            group (str): Optional logical grouping of the parameters
        Returns:
            an Boolean
        """
        value = random.choice([False, True])
        self._record_current_hyperparameters(name, value, group)
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
        value = random.choice(selection)
        if isinstance(selection[0], int):
            value = int(value)
        elif isinstance(selection[0], float):
            value = float(value)
        elif isinstance(selection[0], str):
            value = str(value)
        else:
            Exception('unknown type')
        self._record_current_hyperparameters(name, value, group)
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
        my_range = range(start, stop, increment)
        value = random.choice(my_range)
        self._record_current_hyperparameters(name, value, group)
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
        my_range = logspace(start, stop, num_buckets)
        value = random.choice(my_range)
        self._record_current_hyperparameters(name, value, group)
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
        my_range = linspace(start, stop, num_buckets)
        value = random.choice(my_range)
        if precision:
            value = round(value, precision)
        self._record_current_hyperparameters(name, value, group)
        return value
