from __future__ import absolute_import

from time import time

from .state import State
from kerastuner.collections.metriccollection import MetricsCollection
from kerastuner.abstractions.display import fatal, subsection, display_settings


class ExecutionState(State):
    "Instance Execution state"

    def __init__(self, max_epochs, metrics_config):
        super(ExecutionState, self).__init__()

        self.start_time = int(time())
        self.idx = self.start_time
        self.max_epochs = max_epochs
        self.epochs = 0
        self.eta = -1

        # sub component
        self.metrics = MetricsCollection.from_config(metrics_config)

    def to_config(self):
        self._compute_eta()
        attrs = ['start_time', 'idx', 'epochs', 'max_epochs', 'eta']
        config = self._config_from_attrs(attrs)
        config['record_time'] = int(time())
        config['metrics'] = self.metrics.to_config()
        return config

    @staticmethod
    def from_config(config):
        state = ExecutionState(config["max_epochs"], config["metrics"])

        for attr in ['start_time', 'idx', 'epochs', 'eta']:
            setattr(state, attr, config[attr])
        return state

    def summary(self, extended=False):
        # FIXME: summary
        pass

    def _compute_eta(self):
        "compute remaing time for current model"
        elapsed_time = int(time()) - self.start_time
        time_per_epoch = elapsed_time / max(self.epochs, 1)
        self.eta = int(self.max_epochs * time_per_epoch)
