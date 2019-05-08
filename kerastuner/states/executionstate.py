from __future__ import absolute_import

from time import time

from .state import State
from kerastuner.collections.metricscollection import MetricsCollection
from kerastuner.abstractions.display import fatal, subsection, display_settings

_MAIN_ATTRIBUTES = ['start_time', 'idx', 'epochs', 'max_epochs', 'eta']
_CLASSIFICATION_METRIC_ATTRIBUTES = [
    "confusion_matrix", "roc_curve", "roc_auc_score", "classification_metrics",
    "classifier_type"
]


class ExecutionState(State):
    "Instance Execution state"

    def __init__(self, max_epochs, metrics_config):
        super(ExecutionState, self).__init__()

        self.start_time = int(time())
        self.idx = self.start_time
        self.max_epochs = max_epochs
        self.epochs = 0
        self.eta = -1

        self.confusion_matrix = None
        self.roc_curve = None
        self.roc_auc_score = None
        self.classifier_type = None
        self.classification_metrics = None

        # One-example inference latency
        self.one_example_inference_latency = None

        # sub component
        self.metrics = MetricsCollection.from_config(metrics_config)

    def update_performance_metrics(self, report):
        self.confusion_matrix = report.get("confusion_matrix", None)
        self.roc_curve = report.get("roc_curve", None)
        self.roc_auc_score = report.get("roc_auc_score", None)
        self.classification_metrics = report.get("classification_metrics", None)
        self.classifier_type = report.get("target_type", None)

    def to_config(self):
        self._compute_eta()
        config = self._config_from_attrs(_MAIN_ATTRIBUTES)
        config['record_time'] = int(time())
        config['metrics'] = self.metrics.to_config()

        cfg = self._config_from_attrs(_CLASSIFICATION_METRIC_ATTRIBUTES)
        config['evaluation_metrics'] = cfg

        return config

    @staticmethod
    def from_config(config):
        state = ExecutionState(config["max_epochs"], config["metrics"])

        for attr in _MAIN_ATTRIBUTES:
            setattr(state, attr, config[attr])

        for attr in _CLASSIFICATION_METRIC_ATTRIBUTES:
            setattr(state, attr, config["evaluation_metrics"][attr])

        return state

    def summary(self, extended=False):
        # FIXME: summary
        pass

    def _compute_eta(self):
        "compute remaing time for current model"
        elapsed_time = int(time()) - self.start_time
        time_per_epoch = elapsed_time / max(self.epochs, 1)
        self.eta = int(self.max_epochs * time_per_epoch)
