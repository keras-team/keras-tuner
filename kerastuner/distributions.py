"Function that returns values from a set using specific set of distribution"
from numpy import random, linspace
hyper_parameters = {}


def clear_hyper_parameters():
    "clear the hyperparmeter stack"
    global hyper_parameters
    hyper_parameters = {}


def get_hyper_parameters():
    """Get hyper_parmeters config
    returns
        dict: hyper_parameters dict
    """
    global hyper_parameters
    return hyper_parameters


def __record_parameter(name, value, param_type,  space_size, group):
    """ Record hyper parameters value
    Args:
        name (str): name of the hyperparameter
        value: value of the hyperparameter
        param_type (str): type of hyperparameters
        space_size (int): how big is the param size search space
        group (str): which logical group this parameters belongs to
    """
    global hyper_parameters
    k = "%s:%s" % (group, name)
    if k in hyper_parameters:
        msg = "duplicate name/group for distribution:%s" % k
        raise ValueError(msg)
    hyper_parameters[k] = {"value": "%s" % value, 
                           "type": param_type,
                           "group": group,
                           "space_size": space_size}


def Fixed(name, value, group="default"):
    """Return a fixed selected value
    Args:
        name (str): name of the parameter
        value: value of the parameter
        group (str): Optional logical grouping of the parameters
    Returns:
        value
    """
    __record_parameter(name, value, "Fixed", 1, group)
    return value


def Boolean(name, group="default"):
    """Return a random Boolean value.
    Args:
        name (str): name of the parameter
        group (str): Optional logical grouping of the parameters
    Returns:
        an boolean
    """
    value = random.choice([True, False])
    __record_parameter(name, value, "Boolean", 2, group)
    return value


def Choice(name, selection, group="default"):
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
    __record_parameter(name, value, "Choice", len(selection), group)
    return value


def Range(name, start, stop, increment=1, group='default'):
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
    __record_parameter(name, value, "Range", len(my_range), group)
    return value


def Linear(name, start, stop, num_buckets, precision=0, group='default'):
    """Return a random value from a range which is linearly divided.
    Args:
        name (str): name of the parameter
        start (int/float): lower bound of the range
        stop (int/float): upper bound of the range
        divider (int): into how many buckets should the range being divided in
        precision (int): For float range. Round the result rounded to the
                         nth decimal if needed. 0 means not rounded
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
    __record_parameter(name, value, "Linear", num_buckets, group)
    return value