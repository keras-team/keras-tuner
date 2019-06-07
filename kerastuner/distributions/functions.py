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

from kerastuner.distributions import DummyDistributions
from kerastuner import config

# set dummydistribution as default to allows evaluation and analysis
config._DISTRIBUTIONS = DummyDistributions()


def reset_distributions():
    """Reset the _DISTRIBUTIONS global object to the default."""
    config._DISTRIBUTIONS = DummyDistributions()


def Fixed(name, value, group="default"):
    """Return a fixed selected value
    Args:
        name (str): name of the parameter
        value: value of the parameter
        group (str): Optional logical grouping of the parameters
    Returns:
        value
    """
    return config._DISTRIBUTIONS.Fixed(name, value, group)


def Boolean(name, group="default"):
    """Return a random Boolean value.
    Args:
        name (str): name of the parameter
        group (str): Optional logical grouping of the parameters
    Returns:
        an boolean
    """
    return config._DISTRIBUTIONS.Boolean(name, group)


def Choice(name, selection, group="default"):
    """Return a random value from an explicit list of choice.
    Args:
        name (str): name of the parameter
        selection (list): list of explicit choices
        group (str): Optional logical group name this parameter belongs to.
        Default to 'default'
    Returns:
        an element of the list provided
    """
    return config._DISTRIBUTIONS.Choice(name, selection, group)


def Range(name, start, stop, increment=1, group='default'):
    """Return a random value from a range.
    Args:
        name (str): name of the parameter
        start (int): lower bound of the range
        stop (int): upper bound of the range
        increment (int): incremental step
        group (str): Optional logical group name this parameter belongs to.
        Default to 'default'
    Returns:
        an element of the range
    """
    return config._DISTRIBUTIONS.Range(name, start, stop, increment, group)


def Linear(name, start, stop, num_buckets, precision=0, group='default'):
    """Return a random value from a range which is linearly divided.
    Args:
        name (str): name of the parameter
        start (int/float): lower bound of the range
        stop (int/float): upper bound of the range
        num_buckets (int): in how many buckets to divided the range i
        precision (int): For float range. Round the result rounded to the
                         nth decimal if needed. 0 means not rounded
        group (str): Optional logical group name this parameter belongs to.
        Default to 'default'
    Returns:
        an element of the range
    """
    return config._DISTRIBUTIONS.Linear(name, start, stop, num_buckets,
                                        precision, group)


def Logarithmic(name, start, stop, num_buckets, precision=0, group='default'):
    """Return a random value from a range which is logarithmically divided.
    Args:
        name (str): name of the parameter
        start (int/float): lower bound of the range
        stop (int/float): upper bound of the range
        num_buckets (int): in how many buckets to divided the range in
        precision (int): For float range. Round the result rounded to the
                        nth decimal if needed. 0 means not rounded
        group (str): Optional logical group name this parameter belongs to.
        Default to 'default'
    Returns:
        an element of the range
    """
    return config._DISTRIBUTIONS.Logarithmic(name, start, stop, num_buckets,
                                             precision, group)
