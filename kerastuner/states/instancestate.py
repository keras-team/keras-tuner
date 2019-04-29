import time
import json
from copy import deepcopy

from .state import State
from kerastuner import config
from kerastuner.abstractions.tensorflow import TENSORFLOW as tf
from kerastuner.abstractions.tensorflow import TENSORFLOW_UTILS as tf_utils
from kerastuner.abstractions.display import display_table, section, subsection
from kerastuner.abstractions.display import display_setting, display_settings


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
        # we use deepcopy to avoid mutation due to tuners that swap models
        self.model_size = tf_utils.compute_model_size(model)
        self.optimizer_config = deepcopy(tf.keras.optimizers.serialize(model.optimizer))  # nopep8
        self.loss_config = deepcopy(tf_utils.serialize_loss(model.loss))
        self.model_config = json.loads(model.to_json())
        self.hyper_parameters = deepcopy(hyper_parameters)
        self.agg_metrics = None

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

        subsection("Hyper parameters")
        table = [["Hyperparameter", "Value"]]
        for k, v in self.hyper_parameters.items():
            table.append([k, v["value"]])
        display_table(table, indent=2)

    def to_config(self):
        attrs = ['start_time', 'idx', 'training_size', 'validation_size',
                 'batch_size', 'model_size', 'optimizer_config', 'loss_config',
                 'model_config', 'hyper_parameters']
        config = self._config_from_attrs(attrs)
        config['executions'] = self.execution_configs
        config['hyper_parameters'] = self.hyper_parameters
        if self.agg_metrics:
            config['aggregate_metrics'] = self.agg_metrics.to_config()
        return config
