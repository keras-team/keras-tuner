from __future__ import absolute_import

from time import time

from kerastuner.abstractions.display import fatal
from kerastuner.abstractions.system import System

from .state import State
from .checkpointstate import CheckpointState


class HypertunerState(State):
    "Track hypertuner state"

    def __init__(self, name, user_info):

        # list attributes that should be exported
        self.exportable_attributes = ['name', 'system', 'start_time',
                                      'checkpoint', 'project', 'user_info']

        if not isinstance(user_info, dict):
            fatal('user_info must be a dictionary')

        self.name = name
        self.user_info = user_info
        self.start_time = int(time())
        self.system = System()

        self.checkpoint = None
        self.project = None

    def init_checkpoint(self, is_enabled, monitor, mode):
        "initialize checkpoint state"
        self.checkpoint = CheckpointState(is_enabled, monitor, mode)
