from __future__ import absolute_import

from time import time

from kerastuner.abstractions.display import fatal
from kerastuner.abstractions.system import System

from .state import State
from .dummystate import DummyState
from .checkpointstate import CheckpointState
from .hypertunerstatsstate import HypertunerStatsState



class HypertunerState(State):
    "Track hypertuner state"

    def __init__(self, name, **kwargs):
        """
        [summary]

        Args:
            name (str): hypertuner_name

            epoch_budget(int): defaults to 100. how many epochs to hypertune
            for

            max_budget(int): defaults to 10. how many epochs to spend at most
            on given model



            user_info(dict): additional user supplied information that will be
            recorded alongside training data

        """

        self.epoch_budget = kwargs.get('epoch_budget', 100)
        self._check_type('epoch_budget', self.epoch_budget, int)
        self.remaining_budget = self.epoch_budget

        self.max_epochs = kwargs.get('max_epochs', 10)
        self._check_type('max_epochs', self.max_epochs, int)

        self.user_info = kwargs.get('user_info', {})
        self._check_type('user_info', self.user_info, dict)


        self.name = name
        self.system = System()
        self.start_time = int(time())

        # sub-states
        self.checkpoint = DummyState()
        self.stats = HypertunerStatsState()

    def init_checkpoint(self, is_enabled, monitor, mode):
        "initialize checkpoint state"
        self.checkpoint = CheckpointState(is_enabled, monitor, mode)

    def to_dict(self):
        res = {}

        # collect base attributes values
        attrs = ['name', 'epoch_budget', 'remaining_budget', 'max_epochs',

                 'user_info',
                 'start_time']
        for attr in attrs:
            res[attr] = getattr(self, attr)

        # collect sub components
        res['stats'] = self.stats.to_dict()
        res['checkpoint'] = self.checkpoint.to_dict()
        res['system'] = self.system.to_dict()
        return res

    def _check_type(self, name, obj, expected_type):
        if not isinstance(obj, expected_type):
            fatal('Invalid type for %s -- expected:%s, got:%s' %
                  (name, expected_type, type(obj)))
