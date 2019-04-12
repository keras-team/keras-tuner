from __future__ import absolute_import
from .hypertunerstate import HypertunerState
from .checkpointstate import CheckpointState


class State(object):
    "Instance state abstraction"

    def __init__(self, partial_state=None):

        self.system = None
        self.hypertuner = None
        self.checkpoint = None

        self.project = None

        self.instance = None
        self.user_info = None
        self.executions = []

    def init_hypertuner(self, tuner_name):
        "initialize hypertuner state"
        self.hypertuner = HypertunerState(tuner_name)

    def init_checkpoint(self, is_enabled, monitor, mode):
        "initialize checkpoint state"
        self.checkpoint = CheckpointState(is_enabled, monitor, mode)