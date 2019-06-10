# Copyright 2019 The Keras Tuner Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
from kerastuner.abstractions.display import warning
from kerastuner.collections.metricscollection import MetricsCollection
from ..collections.executionstatescollection import ExecutionStatesCollection

from .state import State
from .executionstate import ExecutionState


class InstanceState(State):
    # FIXME documentations

    _ATTRS = [
        'start_time', 'idx', 'training_size', 'validation_size', 'batch_size',
        'model_size', 'optimizer_config', 'loss_config', 'model_config',
        'metrics_config', 'hyper_parameters', 'is_best_model', 'objective'
    ]

    def __serialize_metrics(self, metrics):
        if isinstance(metrics, list):
            out = []
            for metric in metrics:
                out.append(self.__serialize_metrics(metric))
            return json.dumps(out)
        elif isinstance(metrics, dict):
            out = {}
            for k, v in metrics.items():
                out[k] = self.__serialize_metrics(v)
            return json.dumps(out)
        elif isinstance(metrics, str):
            return json.dumps(metrics)
        else:
            cfg = tf.keras.metrics.serialize(metrics)
            name = cfg["config"]["name"]
            if name == "acc":
                name = "accuracy"
            return name

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
        self.optimizer_config = deepcopy(
            tf.keras.optimizers.serialize(model.optimizer))  # nopep8
        self.loss_config = deepcopy(tf_utils.serialize_loss(model.loss))
        self.model_config = json.loads(model.to_json())
        self.metrics_config = self.__serialize_metrics(model.metrics)
        self.hyper_parameters = deepcopy(hyper_parameters)
        self.agg_metrics = None
        self.is_best_model = False
        self.objective = None  # needed by tools that only have this state

    def set_objective(self, name):
        "Set tuning objective"
        # leverage metric canonicalization
        self.objective = self.agg_metrics.set_objective(name)

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
    def model_from_configs(model_config,
                           loss_config,
                           optimizer_config,
                           metrics_config,
                           weights_filename=None):
        """Creates a Keras model from the configurations typically stored in
        an InstanceState.

        Args:
            model_config (dict): Configuration dictionary representing the
                model.
            loss_config (dict, list, or str): Configuration representing the
                loss(es) for the model.
            optimizer_config (dict): Configuration representing the optimizer.
            metrics_config (dict, list, or str): Configuration representing the
                metrics.
            weights_filename (str, optional): Filename containing weights to
                load.
        Returns:
            Model: The Keras Model defined by the config objects.
        """

        model = model_from_json(json.dumps(model_config))
        loss = tf_utils.deserialize_loss(loss_config)
        optimizer = tf.keras.optimizers.deserialize(optimizer_config)  # nopep8
        metrics = json.loads(metrics_config)
        model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

        if weights_filename:
            if tf.io.gfile.exists(weights_filename):
                model.load_weights(weights_filename)
            else:
                warning("No weights file: '%s'" % weights_filename)

        return model

    @staticmethod
    def model_from_config(config, weights_filename=None):
        """Creates a Keras Model based on an an InstanceState config
        dictionary.

        Args:
            config (dict): InstanceState config, as returned by to_config()
            weights_filename (str, optional): Filename containing weights to load.
        Returns:
            Model: The Keras Model for the configuration.
        """
        model = InstanceState.model_from_configs(
            config['model_config'],
            config['loss_config'],
            config['optimizer_config'],
            config.get('metrics_config', 'null'), 
            weights_filename)

        return model

    def recreate_model(self, weights_filename=None):
        """Recreates the model configured for this instance.

        Returns:
            Model: The Keras Model for this InstanceState.
        """
        model = InstanceState.model_from_configs(self.model_config,
                                                 self.loss_config,
                                                 self.optimizer_config,
                                                 self.metrics_config,
                                                 weights_filename)
        return model

    @staticmethod
    def from_config(config):
        idx = config['idx']
        hyper_parameters = config['hyper_parameters']
        model = InstanceState.model_from_config(config)
        state = InstanceState(idx, model, hyper_parameters)
        for attr in InstanceState._ATTRS:
            setattr(state, attr, config.get(attr, None))

        state.execution_states_collection = (
            ExecutionStatesCollection.from_config(config["executions"]))

        state.agg_metrics = MetricsCollection.from_config(
            config['aggregate_metrics'])  # nopep8

        # !canonicalize objective name
        m = state.agg_metrics.get(state.objective)
        state.objective = m.name

        return state
