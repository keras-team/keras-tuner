"Function that returns values from a set using specific set of distribution"
from numpy import random, linspace
hyper_parameters = {}

def __record_parameter(name, value, param_type):
    global hyperparameters
    hyper_parameters[name] = {"value": "%s" % value, "type": param_type}

def Fixed(name, value):
    "Return a fixed selected value"
    __record_parameter(name, value, "Fixed")
    return value


def Boolean(name):
    """Return a random Boolean value.
    Args:
        name (str): name of the parameter
    Returns:
        an boolean
    """
    value = random.choice([True, False])
    __record_parameter(name, value, "Boolean")
    return value

def Choice(name, selection):
    """Return a random value from an explicit list of choice.
    Args:
        name (str): name of the parameter
        selection (list): list of explicit choices
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
    __record_parameter(name, value, "Choice")
    return value


def Range(name, start, stop, increment=1):
    """Return a random value from a range.
    Args:
        name (str): name of the parameter
        start (int): lower bound of the range
        stop (int): upper bound of the range
        increment (int): incremental step
    Returns:
        an element of the range
    """
    my_range = range(start, stop, increment)
    value = int(random.choice(my_range))
    __record_parameter(name, value, "Range")
    return value


def Linear(name, start, stop, num_buckets, precision=0):
    """Return a random value from a range which is linearly divided.
    Args:
        name (str): name of the parameter
        start (int/float): lower bound of the range
        stop (int/float): upper bound of the range
        divider (int): into how many buckets should the range being divided in
        precision (int): For float range. Round the result rounded to the nth decimal if needed. 0 means not rounded
    Returns:
        an element of the range
    """
    my_range = linspace(start, stop, num_buckets)
    value = random.choice(my_range)
    if isinstance(start, int):
        value = int(value)
    else:
        value = float(value)
        if precision > 0:
            value = round(value, precision + 1)
    __record_parameter(name, value, "Linear")
    return value