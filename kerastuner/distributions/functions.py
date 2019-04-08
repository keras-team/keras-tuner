from kerastuner.distributions import DummyDistributions

DISTRIBUTIONS = DummyDistributions()  # global distrubution object


def Fixed(name, value, group="default"):
    """Return a fixed selected value
    Args:
        name (str): name of the parameter
        value: value of the parameter
        group (str): Optional logical grouping of the parameters
    Returns:
        value
    """
    return DISTRIBUTIONS.Fixed(name, value, group)


def Boolean(name, group="default"):
    """Return a random Boolean value.
    Args:
        name (str): name of the parameter
        group (str): Optional logical grouping of the parameters
    Returns:
        an boolean
    """
    return DISTRIBUTIONS.Boolean(name, group)


def Choice(name, selection, group="default"):
    """Return a random value from an explicit list of choice.
    Args:
        name (str): name of the parameter
        selection (list): list of explicit choices
        group (str): Optional logical group name this parameter belongs to
    Returns:
        an element of the list provided
    """
    return DISTRIBUTIONS.Choice(name, selection, group)


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
    return DISTRIBUTIONS.Range(name, start, stop, increment, group)


def Linear(name, start, stop, num_buckets, precision=0, group='default'):
    """Return a random value from a range which is linearly divided.
    Args:
        name (str): name of the parameter
        start (int/float): lower bound of the range
        stop (int/float): upper bound of the range
        num_buckets (int): in how many buckets to divided the range i
        precision (int): For float range. Round the result rounded to the
                         nth decimal if needed. 0 means not rounded
    Returns:
        an element of the range
    """
    return DISTRIBUTIONS.Linear(name, start, stop, num_buckets,
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
    Returns:
        an element of the range
    """
    return DISTRIBUTIONS.Logarithmic(name, start, stop, num_buckets,
                                     precision, group)
