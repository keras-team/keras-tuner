from abc import abstractmethod
import numpy as np

class Distributions(object):
    """Distributions abstract class"""

    def __init__(self, name):
        self.name = name
        hyperparameters_config = {}
        self.hyperparameters_config = hyperparameters_config
        self.current_hyperparameters = {}  # hparams of the current instance

    @abstractmethod
    def Fixed(self, name, value, group="default"):
        """Return a fixed selected value
        Args:
            name (str): name of the parameter
            value: value of the parameter
            group (str): Optional logical grouping of the parameters
        Returns:
            fixed value
        """

    @abstractmethod
    def Boolean(self, name, group="default"):
        """Return a random Boolean value.
        Args:
            name (str): name of the parameter
            group (str): Optional logical grouping of the parameters
        Returns:
            a boolean
        """

    @abstractmethod
    def Choice(self, name, selection, group="default"):
        """Return a random value from an explicit list of choice.
        Args:
            name (str): name of the parameter
            selection (list): list of explicit choices
            group (str): Optional logical group name this parameter belongs to
        Returns:
            an element of the list provided
        """
        pass

    @abstractmethod
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

    @abstractmethod
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

    @abstractmethod
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
        pass

    def get_current_hyperparameters(self):
        "Return current concretize values"
        return self.current_hyperparameters

    def _record_current_hyperparameters(self, name, value, group):
        """ Record hyperparameters value
        Args:
            name (str): name of the hyperparameter
            value: value of the hyperparameter
            group (str): which logical group this parameters belongs to
        """

        hparam = {
            "name": name,
            "value": value,
            "group": group
        }
        key = self._get_key(name, group)
        self.current_hyperparameters[key] = hparam

    def _get_key(self, name, group):
        """ Generate hyperparameter dict key
        Args:
            name (str): name of the hyperparameter
            group (str): which logical group this parameters belongs to
        Returns:
            str: dict key
        """
        return "%s:%s" % (group, name)

    # hyperparams related function
    def get_hyperparameters_config(self):
        """Get hyper_parmeters config
        returns
            dict: hyperparameters dict
        """
        return self.hyperparameters_config

    def set_hyperparameters_config(self, hyperparameters_config):
        """Set hyperparameter config

        Args:
            hyperparameters_config (dict): dict containing the hyperparams conf
        """
        self.hyperparameters_config = hyperparameters_config
