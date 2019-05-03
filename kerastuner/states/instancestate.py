import json
import time
from collections import defaultdict
from copy import deepcopy

from tensorflow.keras.models import model_from_json  # nopep8 pylint: disable=import-error

from kerastuner import config
from kerastuner.abstractions.display import colorize, colorize_row
from kerastuner.abstractions.display import display_setting, display_settings
from kerastuner.abstractions.display import display_table, section
from kerastuner.abstractions.display import subsection
from kerastuner.abstractions.tensorflow import TENSORFLOW as tf
from kerastuner.abstractions.tensorflow import TENSORFLOW_UTILS as tf_utils
from kerastuner.abstractions.tf import compute_model_size
from kerastuner.abstractions.io import get_weights_filename
from kerastuner.collections.metriccollection import MetricsCollection
from ..collections.executionstatescollection import ExecutionStatesCollection

from .state import State
from .executionstate import ExecutionState


class InstanceState(State):
    # FIXME documentations

    _ATTRS = ['start_time', 'idx', 'training_size', 'validation_size',
              'batch_size', 'model_size', 'optimizer_config', 'loss_config',
              'model_config', 'hyper_parameters', 'is_best_model',
              'objective']

    def __init__(self, idx, model, hyper_parameters):
        super(InstanceState, self).__init__()
        self.start_time = int(time.time())
        self.idx = idx

        # training info
        self.training_size = -1
        self.validation_size = -1
        self.batch_size = -1
        self.execution_trained = 0
        self.execution_states_collection = ExecutionStatesCollection()

        # model info
        # we use deepcopy to avoid mutation due to tuners that swap models
        self.model_size = tf_utils.compute_model_size(model)
        self.optimizer_config = deepcopy(tf.keras.optimizers.serialize(model.optimizer))  # nopep8
        self.loss_config = deepcopy(tf_utils.serialize_loss(model.loss))
        self.model_config = json.loads(model.to_json())
        self.hyper_parameters = deepcopy(hyper_parameters)
        self.agg_metrics = None
        self.is_best_model = False
        self.objective = None  # needed by tools that only have this state

    def set_objective(self, name):
        "Set tuning objective"
        self.agg_metrics.set_objective(name)
        self.objective = name

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
        config = self._config_from_attrs(self._ATTRS)
        config['executions'] = self.execution_states_collection.to_config()
        config['hyper_parameters'] = self.hyper_parameters
        if self.agg_metrics:
            config['aggregate_metrics'] = self.agg_metrics.to_config()
        return config

    @staticmethod
    def _model_from_configs(model_config, loss_config, optimizer_config):
        """Creates a Keras model from the configurations typically stored in
        an InstanceState.

        Args:
            model_config (dict): Configuration dictionary representing the
                model.
            loss_config (dict, list, or str): Configuration representing the
                loss(es) for the model.
            optimizer_config (dict): Configuration representing the optimizer.

        Returns:
            Model: The Keras Model defined by the config objects.
        """

        model = model_from_json(json.dumps(model_config))
        model.loss = tf_utils.deserialize_loss(loss_config)
        model.optimizer = tf.keras.optimizers.deserialize(optimizer_config)  # nopep8
        return model

    @staticmethod
    def model_from_config(config):
        """Creates a Keras Model based on an an InstanceState config
        dictionary.

        Args:
            config (dict): InstanceState config, as returned by to_config()

        Returns:
            Model: The Keras Model for the configuration.
        """
        return InstanceState._model_from_configs(
            config['model_config'], config['loss_config'], config['optimizer_config'])

    def recreate_model(self):
        """Recreates the model configured for this instance.

        Returns:
            Model: The Keras Model for this InstanceState.
        """
        return InstanceState._model_from_configs(
            self.model_config, self.loss_config, self.optimizer_config)

    @staticmethod
    def from_config(config):
        idx = config['idx']
        hyper_parameters = config['hyper_parameters']
        model = InstanceState.model_from_config(config)
        state = InstanceState(idx, model, hyper_parameters)
        for attr in InstanceState._ATTRS:
            setattr(state, attr, config[attr])

        state.execution_states_collection = (
            ExecutionStatesCollection.from_config(config["executions"]))

        state.agg_metrics = MetricsCollection.from_config(config['aggregate_metrics'])  # nopep8

        # !canonicalize objective name
        m = state.agg_metrics.get(state.objective)
        state.objective = m.name

        return state
