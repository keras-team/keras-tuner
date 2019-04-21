from .collections import Collection
from kerastuner.engine.metric import Metric
from kerastuner.abstractions.display import warning, fatal

_METRIC_DIRECTION = {
    'acc': 'max',
    'accuracy': 'max',
    'binary_accuracy': 'max',
    'categorical_accuracy': 'max',
    'loss': 'min',
    'sparse_categorical_accuracy': 'max',
    'sparse_top_k_categorical_accuracy': 'max',
    'top_k_categorical_accuracy': 'max',
}


class MetricsCollection(Collection):

    def __init__(self):
        super(MetricsCollection, self).__init__()
        self._objective_name = None  # track which metric is the objective

    def add(self, metric):
        """ Add a metric to the collection

        Args:
            metric (Metric or str): Metric object or metric name
        """

        if isinstance(metric, str):
            metric_name = metric
            no_val_name = metric_name.replace('val_', '')
            if no_val_name in _METRIC_DIRECTION:
                direction = _METRIC_DIRECTION[no_val_name]
            else:
                fatal('Unknown metric %s. Use a custom one?' % metric_name)
            metric = Metric(metric_name, direction)
        else:
            metric_name = metric.name

        if metric_name in self._objects:
            fatal('Duplicate metric:%s' % metric_name)
        self._objects[metric_name] = metric
        self._last_insert_idx = metric_name

    def update(self, metric_name, value):
        """
        Update a given metric

        Args:
            metric_name (str): Name of the metric to update.
            value (float or int): Updated value.

        Returns:
            bool: True if the metric improved, False otherwise.
        """
        metric = self.get(metric_name)
        if metric:
            return metric.update(value)
        return False

    def to_config(self):
        """Serializable list of metrics.

        Returns:
            list: Collection of metric dict
        """

        names = sorted(self._objects.keys())
        # for each metric returns its serialized form
        return [self._objects[name].to_config() for name in names]

    def to_dict(self):
        """ Serializable dict representation of the metric collection.
        Returns:
            list: Collection of metric dict.
        """
        res = {}
        for name, metric in self._objects.items():
            res[name] = metric.to_config()
        return res

    def set_objective(self, name):
        "Mark a metric as tuning objective"
        if name not in self._objects:
            fatal("can't find objective: %s in metric list" % name)
        if self._objective_name:
            fatal("Objective already set to %s" % self._objective_name)
        self._objective_name = name
        self._objects[name].is_objective = True

    def get_objective(self):
        "Get metric objective"
        if not self._objective_name:
            warning("objective not set yet. returning None")
            return None
        return self._objects[self._objective_name]
