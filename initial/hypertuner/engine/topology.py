import random
import copy
class MLayer(object):
    """Abstract meta base layer class."""
    def __init__(self, **kwargs):
        self.optional = kwargs.get('optional', False)
        self.instances = [] # contains all the instances

    def get_random_instance(self):
        "Return an instance of the layer at random"
        #WARNING: You MUST do a deepcopy
        return copy.deepcopy(random.choice(self.instances))