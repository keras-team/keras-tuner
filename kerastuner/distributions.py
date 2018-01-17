"Function that returns values from a set using specific set of distribution"
from numpy import random

def Fixed(val):
  "Return a fixed selected value"
  return val

def Bool():
    "Return a random bool"
    return random.choice([True, False])

def Choice(selection):
    """Return a random value from an explicit list of choice.
    Args:
        selection (list): a list that contains explictly all the choices
    Returns:
        an element of the list
    """
    return random.choice(selection)

def Range(start, stop, increment=1):
    """Return a random value from a range.
    Args:
        start (int/float): lower bound of the range
        stop (int/float): upper bound of the range
        increment (int, float): incremental step
    Returns:
        an element of the range
    
    Todo:
      Don't generate the full range, do something more optimal
    """
    my_range = range(start, stop, increment)
    return random.choice(my_range)

