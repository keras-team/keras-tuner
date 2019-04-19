from __future__ import absolute_import

from time import time

from kerastuner.abstractions.display import fatal

from .state import State
from .dummystate import DummyState
from .checkpointstate import CheckpointState
from .tunerstatsstate import TunerStatsState
from .hoststate import HostState


class TunerState(State):
    """
    Keep track tuner state

    Args:
        name (str): tuner name.

        objective (Objective): Which objective the tuner is optimizing for

        epoch_budget (int): defaults to 100. how many epochs to hypertune
        for

        max_budget (int): defaults to 10. how many epochs to spend at most
        on given model

        min_budget (int): defaults to 3. how many epochs to spend at least
        on given model

        project (str):  defaults to 'default'. project the tuning belong to

        architecture (str): default to timestamp. name of the architecture
        tuned

        user_info(dict): additional user supplied information that will be
        recorded alongside training data

        num_executions(int): defaults to 1. number of execution per model

        dry_run(bool): defaults to False. Run the tuner without training
        models

        debug(bool): defaults to False. Display debug information if true

        display_model(bool): defaults to False. Display model summary if
        true
    """

    def __init__(self, name, objective, **kwargs):
        super(TunerState, self).__init__(**kwargs)

        self.name = name
        self.start_time = int(time())

        # budget
        self.epoch_budget = self._register('epoch_budget', 100, True)
        self.max_epochs = self._register('max_epochs', 10, True)
        self.min_epochs = self._register('min_epochs', 3, True)
        self.remaining_budget = self.epoch_budget

        # user info
        self.project = self._register('project', 'default')
        self.architecture = self._register('architecture', str(time()))
        self.user_info = self._register('user_info', {})

        # execution
        self.num_executions = self._register('num_executions', 1, True)
        self.hyper_parameters = None  # set in Tuner._check_and_store_model_fn
        self.max_parameters = self._register('max_parameters', 50000000)

        # debug
        self.dry_run = self._register('dry_run', False)
        self.debug = self._register('debug', False)
        self.display_model = self._register('display_model', '')

        # sub-states
        self.host = HostState(**kwargs)
        self.checkpoint = DummyState()
        self.stats = TunerStatsState()
        self.checkpoint = CheckpointState(**kwargs)

    def to_config(self):
        res = {}

        # collect user params
        for name in self.user_parameters:
            res[name] = getattr(self, name)

        # collect programtically defined params
        attrs = ['name', 'start_time', 'remaining_budget', 'hyper_parameters']
        for attr in attrs:
            res[attr] = getattr(self, attr)

        # collect sub components
        res['stats'] = self.stats.to_config()
        res['checkpoint'] = self.checkpoint.to_config()
        res['host'] = self.host.to_config()
        return res
