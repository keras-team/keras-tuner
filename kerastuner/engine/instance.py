import json
import time
from collections import defaultdict
from os import path
import tensorflow as tf

from .execution import Execution
from .metric import Metric
from kerastuner.states import InstanceState
from kerastuner.collections import ExecutionsCollection, MetricsCollection
from kerastuner.abstractions.display import section, subsection, fatal


class Instance(object):
    """Model instance class."""

    def __init__(self, idx, model, hparams, tuner_state, cloudservice):

        self.model = model
        self.tuner_state = tuner_state
        self.cloudservice = cloudservice
        self.executions = ExecutionsCollection()

        # init instance state
        self.state = InstanceState(idx, model, hparams)
        self.metrics_config = None  # metric config passed to each executions

    def summary(self, extended=False):
        section("Instance summary")
        self.state.summary(extended=extended)

    def resume_fit(self, fixme):
        """resume fiting an instance
        use execution id?
        """
        pass

    def fit(self, x, y, epochs, **kwargs):
        """Fit an execution of the model instance

        Args:
            x (numpy array): Training data
            epochs (int): Number of epochs to train the model.

        Returns:
            Instance: Instance object
        """

        # collect batch_size from the fit function
        self.state.batch_size = kwargs.get('batch_size', 32)

        # compute training_size and validation_size
        # in theory for batch training the function is __len__
        # should be implemented. However, for generator based training, __len__
        # returns the number of batches, NOT the training size.
        if isinstance(x, tf.keras.utils.Sequence):
            # FIXME: the +2 seems weird but seemed to matter on some testing
            self.state.training_size = (len(x) + 2) * self.state.batch_size
        else:
            self.state.training_size = len(x)

        # Determine the validation size for the various validation strategies.
        if kwargs.get('validation_data'):
            self.state.validation_size = len(kwargs['validation_data'][1])
        elif kwargs.get('validation_split'):
            validation_split = kwargs.get('validation_split')
            val_size = self.state.training_size * validation_split
            self.state.validation_size = val_size
            self.state.training_size -= self.state.validation_size
        else:
            self.state.validation_size = 0
        self.state.validation_size = int(self.state.validation_size)
        self.state.training_size = int(self.state.training_size)

        # init metrics if needed
        if not self.state.agg_metrics:
            self.state.agg_metrics = MetricsCollection()

            # model metrics
            for metric in self.model.metrics:
                self.state.agg_metrics.add(metric)
                if self.state.validation_size:
                    # assume keras metric is printable - might be wrong
                    if not isinstance(metric, str):
                        metric_name = metric.name
                    else:
                        metric_name = metric
                    val_metric = "val_%s" % metric_name
                    self.state.agg_metrics.add(val_metric)

            # loss(es) - model.loss in {str, dict, list}
            if isinstance(self.model.loss, dict):
                losses = list(self.model.loss.keys())
            elif isinstance(self.model.loss, str):
                losses = ['loss']  # single loss is always named loss
            else:
                losses = self.model.loss

            for loss in losses:
                self.state.agg_metrics.add(Metric(loss, 'min'))
                if self.state.validation_size:
                    if not isinstance(loss, str):
                        loss_name = loss.name  # nopep8 pylint: disable=no-member
                    else:
                        loss_name = loss
                    val_loss = "val_%s" % loss_name
                    self.state.agg_metrics.add(Metric(val_loss, 'min'))

            # mark objective
            self.state.set_objective(self.tuner_state.objective)
            self.metrics_config = self.state.agg_metrics.to_config()

            # init tuner global metric if needed (first training)
            if not self.tuner_state.agg_metrics:
                self.tuner_state.agg_metrics = MetricsCollection.from_config(self.metrics_config)  # nopep8

        execution = Execution(self.model, self.state, self.tuner_state,
                              self.metrics_config, self.cloudservice)
        self.executions.add(execution.state.idx, execution)
        execution.fit(x, y, epochs=epochs, **kwargs)
        self.state.execution_trained += 1

        return execution

    def get_best_execution(self):

        objective = self.agg_metrics.get_objective()

        def objective_sort_key(_, execution):
            execution_metrics = execution.state.agg_metrics
            metric = execution_metrics.get(objective.name).get_best_value()
            return metric

        def sort_fn(idx, object):
            return object.state.agg_metrics[objective.name]

        for execution in self.executions.to_list():
            value = ex.state.metrics.get(
                self.state.agg_metrics.objective.name).get_last_value()
