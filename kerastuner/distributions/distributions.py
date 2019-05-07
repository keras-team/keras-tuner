from abc import abstractmethod
import numpy as np
from collections import defaultdict

from kerastuner.abstractions.display import display_table, subsection
from kerastuner.abstractions.display import display_setting, fatal, warning
from kerastuner.abstractions.display import colorize, colorize_row


class Distributions(object):
    """Distributions abstract class
    Args:
        name (str): name of the distribution

        hyperparameters_config (dict): hyperparameters dict describing
        the search space. Often refered as hparams. Generated using
        DummyDistributions() in Tuner()

        fatal_on_dynamic_hyperparmeter (Bool): Raise a ValueError when
        a hyperparmeter not in the config is being recorded. Need to be set
        for Distribution or hypertuner algorithm that requires params to be
        known in advance. Defaults to False.

    Attributes:
        _hyperparameters_config (dict): hparams object describing search space
        _hyperparameters (dict): current set of selected parameters

    """

    def __init__(self, name, hyperparameters_config,
                 fatal_on_dynamic_hyperparmeter=False):
        self.name = name
        self._hyperparameters_config = hyperparameters_config
        self._hyperparameters = {}  # hparams of the current instance
        self.dynamic_hyperparameters = False
        self.fatal_on_dynamic_hyperparmeter = fatal_on_dynamic_hyperparmeter

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

    # current hyperparameters related functions
    def get_hyperparameters(self):
        "Return current hyperparameters config"
        return self._hyperparameters

    def _record_hyperparameter(self, name, value, group):
        """ Record hyperparameter value
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

        # new hyper-parameter - makes reporting unstable
        if key not in self._hyperparameters_config:
            self.dynamic_hyperparameters = True
            if self.fatal_on_dynamic_hyperparmeter:
                fatal('Parameter %s is dynamic - this is incompatible with\
                      tuning algorithm' % key)
            else:
                warning('Parameter %s is dynamic - this will make reporitng\
                        innacurate. Consider making hyperparameters\
                        non-conditional' % key)

        self._hyperparameters[key] = hparam

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
        return self._hyperparameters_config

    def get_search_space_size(self):
        total_size = 1
        for data in self._hyperparameters_config.values():
            total_size *= data['space_size']
        return total_size

    def config_summary(self):
        subsection("Hyper-parmeters search space")
        display_setting("distribution type: %s" % self.name)
        # Compute the size of the hyperparam space by generating a model
        total_size = 1
        data_by_group = defaultdict(dict)
        group_size = defaultdict(lambda: 1)
        for data in self._hyperparameters_config.values():
            data_by_group[data['group']][data['name']] = data['space_size']
            group_size[data['group']] *= data['space_size']
            total_size *= data['space_size']

        # Generate the table.
        rows = [['Hyper-parameter', 'Search space']]
        for grp in sorted(data_by_group.keys()):
            row = ["%s total" % grp, '']
            row = colorize_row(row, 'cyan')
            rows.append(row)

            for param, size in data_by_group[grp].items():
                rows.append(["|-%s" % param, size])

            # rows.append(['', ''])
        rows.append([colorize('total', 'green'),
                     colorize(total_size, 'green')])
        display_table(rows)
