from __future__ import absolute_import

from time import time

from .state import State
from kerastuner.abstractions.display import fatal, subsection, display_settings


class ExecutionState(State):
    "Instance Execution state"

    def __init__(self):
        super(ExecutionState, self).__init__()

        self.start_time = int(time())
        self.num_epoch = -1

    def to_config(self):
        pass

    def summary(self, extended=False):
        pass
