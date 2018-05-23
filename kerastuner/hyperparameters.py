"Store the parameters used for tuning"
from . import distributions

class HyperParameters(object):

    def __init__(self):
        self.parameters = {}

    def Fixed(self, name, value):
        "Return a fixed selected value"
        self.__record_parameter(name, value, "Fixed")
        return value


    def Boolean(self, name):
        """Return a random Boolean value.
        Args:
            name (str): name of the parameter
        Returns:
            an boolean
        """
        value = distributions.Boolean()
        self.__record_parameter(name, value, "Boolean")
        return value

    def Choice(self, name, selection):
        """Return a random value from an explicit list of choice.
        Args:
            name (str): name of the parameter
            selection (list): list of explicit choices
        Returns:
            an element of the list provided
        """
        value = distributions.Choice(selection)
        self.__record_parameter(name, value, "Choice")
        return value


    def Range(self, name, start, stop, increment=1):
        """Return a random value from a range.
        Args:
            name (str): name of the parameter
            start (int): lower bound of the range
            stop (int): upper bound of the range
            increment (int): incremental step
        Returns:
            an element of the range
        """
        value = distributions.Range(start, stop, increment)
        self.__record_parameter(name, value, "Range")
        return value


    def Linear(self, name, start, stop, num_buckets, precision=0):
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
        value = distributions.Linear(start, stop, num_buckets, precision)
        self.__record_parameter(name, value, "Linear")
        return value


    def get_parameters(self):
        return self.parameters

    def __record_parameter(self, name, value, dist_type):
        if name == None or name == '':
            Exception('Please specify a valid name')
        self.parameters[name] = {"name": name, 'type': dist_type}