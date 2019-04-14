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

    def __init__(self, hypertuner_name, **kwargs):
        """
        [summary]

        Args:
            hypertuner_name (str): hypertuner name.

            epoch_budget(int): defaults to 100. how many epochs to hypertune
            for

            max_budget(int): defaults to 10. how many epochs to spend at most
            on given model



            user_info(dict): additional user supplied information that will be
            recorded alongside training data

        """

        super(HypertunerState, self).__init__(**kwargs)

        # user params
        self.epoch_budget = self._register('epoch_budget', 100, True)
        self.max_epochs = self._register('max_epochs', 10, True)
        self.user_info = self._register('user_info', {})

        # system parameters
        self.name = hypertuner_name
        self.start_time = int(time())
        self.remaining_budget = self.epoch_budget

        # sub-states
        self.system = System()
        self.checkpoint = DummyState()
        self.stats = HypertunerStatsState()
        self.checkpoint = CheckpointState(**kwargs)

    def to_dict(self):
        res = {}

        # collect user params
        for name in self.user_parameters:
            res[name] = getattr(self, name)

        # collect programtically defined params
        attrs = ['name', 'start_time', 'remaining_budget']
        for attr in attrs:
            res[attr] = getattr(self, attr)

        # collect sub components
        res['stats'] = self.stats.to_dict()
        res['checkpoint'] = self.checkpoint.to_dict()
        res['system'] = self.system.to_dict()
        return res
