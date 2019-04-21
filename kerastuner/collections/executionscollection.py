import json

from .collections import Collection


class ExecutionsCollection(Collection):
    """ Manage a collection of executions
    """

    def __init__(self):
        super(ExecutionsCollection, self).__init__()
