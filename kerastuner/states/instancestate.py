import time
import json
import tensorflow as tf
from copy import deepcopy
from collections import defaultdict

from .state import State
from kerastuner import config
from kerastuner.abstractions.tf import compute_model_size
from kerastuner.abstractions.io import serialize_loss
from kerastuner.abstractions.display import display_table, section, subsection
from kerastuner.abstractions.display import display_setting, display_settings
from kerastuner.abstractions.display import colorize, colorize_row


class InstanceState(State):
    # FIXME documentations

    def __init__(self, idx, model, hyper_parameters):
        super(InstanceState, self).__init__()
        self.start_time = int(time.time())
        self.idx = idx

        # training info
        self.training_size = -1
        self.validation_size = -1
        self.batch_size = -1
        self.execution_trained = 0
        self.execution_configs = []

        # model info
        # we use deepcopy to avoid mutation due to tuner that swap models
        self.model_size = compute_model_size(model)
        self.optimizer_config = deepcopy(tf.keras.optimizers.serialize(model.optimizer))  # nopep8
        self.loss_config = deepcopy(serialize_loss(model.loss))
        self.model_config = json.loads(model.to_json())
        self.hyper_parameters = deepcopy(hyper_parameters)
        self.agg_metrics = None
        self.is_best_model = False

    def summary(self, extended=False):
        subsection('Training parameters')
        settings = {"idx": self.idx, "model size": self.model_size}
        if extended:
            settings.update({
                "training size": self.training_size,
                "validation size": self.validation_size,
                "batch size": self.batch_size
                })
        display_settings(settings)

        subsection("Hyper-parameters")
        # group params
        data_by_group = defaultdict(dict)
        for data in self.hyper_parameters.values():
            data_by_group[data['group']][data['name']] = data['value']

        # Generate the table.
        rows = [['Group', 'Hyper-parameter', 'Value']]
        idx = 0
        for grp in sorted(data_by_group.keys()):
            for param, value in data_by_group[grp].items():
                row = [grp, param, value]
                if idx % 2:
                    row = colorize_row(row, 'cyan')
                rows.append(row)
                idx += 1
        display_table(rows)

    def to_config(self):
        attrs = ['start_time', 'idx', 'training_size', 'validation_size',
                 'batch_size', 'model_size', 'optimizer_config', 'loss_config',
                 'model_config', 'hyper_parameters', 'is_best_model']
        config = self._config_from_attrs(attrs)
        config['executions'] = self.execution_configs
        config['hyper_parameters'] = self.hyper_parameters
        if self.agg_metrics:
            config['aggregate_metrics'] = self.agg_metrics.to_config()
        return config
