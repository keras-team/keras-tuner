import numpy

def Fixed(val):
    "Return a fixed selected value"
    return [val]

def Choice(selection):
    "Return all possible choices"
    return selection

def Range(min_val, max_val, increment=1):
    "Return all values for a given range"
    return range(min_val, max_val, increment)

def uniform(min_val, max_val, num_samples):
    "Return a uniform set of values over the space"
    numpy.random.uniform(low=min_val, high=max_val, size=num_samples)