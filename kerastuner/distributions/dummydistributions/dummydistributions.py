# Copyright 2019 The Keras Tuner Authors
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     https://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np

from ..distributions import Distributions
from kerastuner.abstractions.display import fatal


class DummyDistributions(Distributions):
    "Dummy distribution class used to record and test hyper parameters space"

    def __init__(self, max_reported_values=100):
        # !DO NOT do a super -- this is the bootstrap class
        self._hyperparameters = {}
        self._hyperparameters_config = {}
        self.max_reported_values = max_reported_values

    def _record_hyperparameters(self, name, htype, space_size, start, stop,
                                group, values):
        """
        Record a given hyperparameter

        Args:
            name (str): name of the hyperparameter
            htype (str): type of hyperparameter
            space_size (int): number of values the param can take
            start: lower bound
            stop: upper bound
            values (list): list of potential values. Truncated to 100
        """

        key = self._get_key(name, group)

        # check if we have a duplicate
        if key in self._hyperparameters_config:
            fatal("%s hyperparameter is declared twice" % key)

        # making sure values are serializable
        serializable_values = []
        for v in values[:self.max_reported_values]:
            if isinstance(v, np.integer):
                serializable_values.append(int(v))
            if isinstance(v, np.float):
                serializable_values.append(float(v))
            else:
                serializable_values.append(v)

        self._hyperparameters_config[key] = {
            "name": name,
            "group": group,
            "type": htype,
            "space_size": space_size,
            "start": start,
            "stop": stop,
            "values": serializable_values
        }

    def Fixed(self, name, value, group="default"):
        """Return a fixed selected value
        Args:
            name (str): name of the parameter
            value: value of the parameter
            group (str): Optional logical grouping of the parameters
        Returns:
            fixed value
        """
        self._record_hyperparameters(name, 'Fixed', 1, value, value, group,
                                     [value])
        return value

    def Boolean(self, name, group="default"):
        """Return a random Boolean value.
        Args:
            name (str): name of the parameter
            group (str): Optional logical grouping of the parameters
        Returns:
            a boolean
        """
        self._record_hyperparameters(name, 'Boolean', 2, True, False, group,
                                     [True, False])
        return True

    def Choice(self, name, selection, group="default"):
        """Return a random value from an explicit list of choice.
        Args:
            name (str): name of the parameter
            selection (list): list of explicit choices
            group (str): Optional logical group name this parameter belongs to
        Returns:
            an element of the list provided
        """
        if not isinstance(selection, list):
            fatal("list if choice must be a list []")

        self._record_hyperparameters(name, 'Choice', len(selection),
                                     selection[0], selection[-1], group,
                                     selection)
        return selection[0]

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
        if not isinstance(start, int) or not isinstance(stop, int):
            fatal("start, stop must be integers")
        if not isinstance(increment, int):
            fatal("increment must be an integer")
        if stop <= start:
            fatal("start value:%s larger than stop value:%s" % (start, stop))

        rsize = stop - start
        if rsize < increment:
            fatal("increment: %s greater than range size:%s" % (increment,
                                                                rsize))

        my_range = list(range(start, stop, increment))
        self._record_hyperparameters(name, 'Range', len(my_range),
                                     start, stop, group, my_range)
        return start

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
        if not isinstance(start, float) and not isinstance(start, int):
            fatal("start must be a float or an int")
        if not isinstance(stop, float) and not isinstance(stop, int):
            fatal("stop must be a float or an int")
        if not isinstance(num_buckets, int):
            fatal("num_bucket must be an integer")
        if stop <= start:
            fatal("start value:%s larger than stop value:%s" % (start, stop))
        my_range = np.logspace(start, stop, num_buckets)
        self._record_hyperparameters(name, 'Logarithmic', num_buckets, start,
                                     stop, group, my_range)
        return start

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
        if not isinstance(start, float) and not isinstance(start, int):
            fatal("start must be a float or an int")
        if not isinstance(stop, float) and not isinstance(stop, int):
            fatal("stop must be a float or an int")
        if not isinstance(num_buckets, int):
            fatal("num_bucket must be an integer")
        if stop <= start:
            fatal("start value:%s larger than stop value:%s" % (start, stop))

        my_range = np.linspace(start, stop, num_buckets)
        self._record_hyperparameters(name, 'Linear', num_buckets, start, stop,
                                     group, my_range)
        return start
